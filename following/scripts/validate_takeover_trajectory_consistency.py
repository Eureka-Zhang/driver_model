#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution and time-series consistency validation: raw calibrated vs generated takeover.

Compares real (source) and generated following trajectories on speed, acceleration,
headway, THW, TTC, inv_TTC, inv_THW, and jerk. Produces per-driver figures, a fleet
summary, and CSV reports for consistency checks:

  1. Headway mean/var close to the source session
  2. Acceleration distribution within acceptable driving range
  3. THW risk metrics without excessive abnormal values (TTC reported only)
  4. Generated relative speed self-consistent with ego/lead speeds; gap geometry self-consistent

Usage::

  python3 following/scripts/validate_takeover_trajectory_consistency.py \
    --generated_dir following/outputs/residual_gru_takeover_20s_yaw_shrink_controls \
    --generation_summary following/outputs/residual_gru_takeover_20s/generation_summary.csv \
    --out_dir following/outputs/pictures/takeover_consistency_yaw_shrink_controls \
    --min_time_s 20.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAW_COLOR = "#1f77b4"
GEN_COLOR = "#ff7f0e"
PASS_COLOR = RAW_COLOR
FAIL_COLOR = GEN_COLOR

METRIC_SPECS: Tuple[Tuple[str, Tuple[str, ...], str, bool], ...] = (
    ("speed", ("ego_speed", "ego_v_long"), "Speed (m/s)", False),
    ("accel", ("ego_acceleration", "ego_a_long"), "Accel (m/s²)", False),
    ("headway", ("distance_headway",), "Headway (m)", False),
    ("thw", ("time_headway",), "THW (s)", True),
    ("ttc", ("ttc",), "TTC (s)", True),
    ("inv_thw", ("inv_time_headway",), "1/THW (1/s)", True),
    ("inv_ttc", ("inv_ttc",), "1/TTC (1/s)", True),
    ("jerk", ("ego_jerk",), "Jerk (m/s³)", False),
)


def _driver_sort_key(name: str) -> Tuple[int, str]:
    suffix = name[1:] if name.startswith("T") else name
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _mask_sentinel(arr: np.ndarray, *, sentinel: float = 999.0, positive_only: bool = False) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    out = x.copy()
    bad = ~np.isfinite(out)
    bad |= np.isclose(out, sentinel, rtol=0.0, atol=1e-3)
    if positive_only:
        bad |= out <= 0.0
    out[bad] = np.nan
    return out


def _series_from_df(df: pd.DataFrame, names: Tuple[str, ...], mask_risk: bool) -> np.ndarray:
    col = None
    for name in names:
        if name in df.columns:
            col = name
            break
    if col is None:
        return np.asarray([], dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if mask_risk:
        vals = _mask_sentinel(vals, positive_only=True)
    return vals


def _stats(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(
            n=0, mean=float("nan"), std=float("nan"), p05=float("nan"),
            p50=float("nan"), p95=float("nan"), min=float("nan"), max=float("nan"),
        )
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x)),
        p05=float(np.percentile(x, 5)),
        p50=float(np.percentile(x, 50)),
        p95=float(np.percentile(x, 95)),
        min=float(np.min(x)),
        max=float(np.max(x)),
    )


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        qs = np.linspace(0.05, 0.95, 19)
        return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _load_generation_sources(summary_csv: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not summary_csv.is_file():
        return out
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            did = str(row.get("driver_id", "")).strip()
            src = str(row.get("source_csv", "")).strip()
            if did and src:
                out[did] = Path(src)
    return out


def _load_df(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    tcol = "sim_time_s" if "sim_time_s" in df.columns else "timestamp"
    df["__t"] = pd.to_numeric(df[tcol], errors="coerce").astype(float)
    return df


def _slice_window(df: pd.DataFrame, min_time_s: Optional[float], max_time_s: Optional[float]) -> pd.DataFrame:
    out = df
    if min_time_s is not None:
        out = out[out["__t"] >= float(min_time_s)]
    if max_time_s is not None:
        out = out[out["__t"] < float(max_time_s)]
    return out.reset_index(drop=True)


def _infer_half_len_sum(row: pd.Series, gap_m: float) -> float:
    try:
        ex, ey = float(row["ego_pos_x"]), float(row["ego_pos_y"])
        lx, ly = float(row["lead_pos_x"]), float(row["lead_pos_y"])
        center = math.hypot(ex - lx, ey - ly)
        half = center - float(gap_m)
        return half if half >= 0.05 else 4.5
    except (TypeError, ValueError, KeyError):
        return 4.5


def _geometry_gap_error(df: pd.DataFrame) -> np.ndarray:
    if not {"ego_pos_x", "ego_pos_y", "lead_pos_x", "lead_pos_y", "distance_headway"}.issubset(df.columns):
        return np.asarray([], dtype=float)
    half = None
    errs = []
    for _, row in df.iterrows():
        dh = pd.to_numeric(row.get("distance_headway"), errors="coerce")
        if not np.isfinite(dh):
            continue
        if half is None:
            half = _infer_half_len_sum(row, float(dh))
        ex, ey = float(row["ego_pos_x"]), float(row["ego_pos_y"])
        lx, ly = float(row["lead_pos_x"]), float(row["lead_pos_y"])
        center = math.hypot(ex - lx, ey - ly)
        geom = center - half
        errs.append(float(dh) - geom)
    return np.asarray(errs, dtype=float)


def _relative_speed_internal_mae(df: pd.DataFrame) -> float:
    """MAE between stored relative speed and lead_speed - ego_speed."""
    ego = _series_from_df(df, ("ego_speed", "ego_v_long"), False)
    lead = _series_from_df(df, ("lead_speed", "lead_v_long"), False)
    rel_col = _series_from_df(df, ("relative_v_long", "relative_speed"), False)
    if ego.size == 0 or lead.size == 0 or rel_col.size == 0:
        return float("nan")
    n = min(len(ego), len(lead), len(rel_col))
    expected = lead[:n] - ego[:n]
    err = rel_col[:n] - expected
    err = err[np.isfinite(err)]
    return float(np.mean(np.abs(err))) if err.size else float("nan")


def _check_flags(
    raw: Dict[str, np.ndarray],
    gen: Dict[str, np.ndarray],
    geom_err: np.ndarray,
    rel_speed_internal_raw: float,
    rel_speed_internal_gen: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    flags: Dict[str, object] = {}
    dh_r, dh_g = raw.get("headway", np.array([])), gen.get("headway", np.array([]))
    sr, sg = _stats(dh_r), _stats(dh_g)
    if sr["mean"] == sr["mean"] and sg["mean"] == sg["mean"]:
        flags["gap_mean_diff_m"] = sg["mean"] - sr["mean"]
        flags["gap_mean_ratio"] = sg["mean"] / sr["mean"] if abs(sr["mean"]) > 1e-6 else float("nan")
    else:
        flags["gap_mean_diff_m"] = float("nan")
        flags["gap_mean_ratio"] = float("nan")
    if sr["std"] == sr["std"] and sg["std"] == sg["std"] and sr["std"] > 1e-6:
        flags["gap_std_ratio"] = sg["std"] / sr["std"]
    else:
        flags["gap_std_ratio"] = float("nan")

    flags["gap_mean_ok"] = (
        np.isfinite(flags["gap_mean_ratio"])
        and abs(float(flags["gap_mean_ratio"]) - 1.0) <= args.gap_mean_ratio_tol
    )
    flags["gap_std_ok"] = (
        np.isfinite(flags["gap_std_ratio"])
        and abs(float(flags["gap_std_ratio"]) - 1.0) <= args.gap_std_ratio_tol
    )

    acc = gen.get("accel", np.array([]))
    acc = acc[np.isfinite(acc)]
    flags["accel_min"] = float(np.min(acc)) if acc.size else float("nan")
    flags["accel_max"] = float(np.max(acc)) if acc.size else float("nan")
    flags["accel_ok"] = (
        acc.size > 0
        and float(flags["accel_min"]) >= args.accel_min_mps2
        and float(flags["accel_max"]) <= args.accel_max_mps2
    )

    for key in ("ttc", "thw"):
        vals = gen.get(key, np.array([]))
        vals = vals[np.isfinite(vals)]
        flags["{}_abnormal_frac".format(key)] = (
            float(np.mean((vals < args.risk_min_s) | (vals > args.risk_max_s))) if vals.size else float("nan")
        )
    flags["ttc_risk_checked"] = False
    flags["risk_ok"] = (
        np.isfinite(flags["thw_abnormal_frac"])
        and float(flags["thw_abnormal_frac"]) <= args.risk_abnormal_frac_tol
    )

    ge = geom_err[np.isfinite(geom_err)]
    flags["geom_gap_mae_m"] = float(np.mean(np.abs(ge))) if ge.size else float("nan")
    flags["rel_speed_internal_mae_raw"] = rel_speed_internal_raw
    flags["rel_speed_internal_mae_gen"] = rel_speed_internal_gen
    flags["geometry_ok"] = (
        np.isfinite(flags["geom_gap_mae_m"])
        and float(flags["geom_gap_mae_m"]) <= args.geom_gap_mae_tol_m
    )
    flags["relative_speed_ok"] = (
        np.isfinite(rel_speed_internal_gen)
        and (
            float(rel_speed_internal_gen) <= args.rel_speed_mae_tol_mps
            or (
                np.isfinite(rel_speed_internal_raw)
                and float(rel_speed_internal_gen) <= float(rel_speed_internal_raw) + args.rel_speed_mae_tol_mps
            )
        )
    )
    flags["all_ok"] = all([
        flags["gap_mean_ok"], flags["gap_std_ok"], flags["accel_ok"],
        flags["risk_ok"], flags["geometry_ok"], flags["relative_speed_ok"],
    ])
    return flags


def _plot_driver(
    driver: str,
    raw: pd.DataFrame,
    gen: pd.DataFrame,
    metric_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    flags: Dict[str, object],
    out_dir: Path,
    takeover_time_s: float,
) -> None:
    import matplotlib.pyplot as plt

    t_r, t_g = raw["__t"].to_numpy(), gen["__t"].to_numpy()
    status = "PASS" if flags.get("all_ok") else "CHECK"
    title_suffix = (
        "{} {} | gap mu ratio={:.3f} sigma ratio={:.3f} | geom MAE={:.3f} m".format(
            driver,
            status,
            float(flags.get("gap_mean_ratio", float("nan"))),
            float(flags.get("gap_std_ratio", float("nan"))),
            float(flags.get("geom_gap_mae_m", float("nan"))),
        )
    )

    fig_ts, axs_ts = plt.subplots(4, 1, figsize=(12, 11), dpi=120, sharex=True)
    ts_pairs = [
        ("speed", "Speed (m/s)"),
        ("accel", "Accel (m/s²)"),
        ("headway", "Headway (m)"),
        ("jerk", "Jerk (m/s³)"),
    ]
    for ax, (key, ylab) in zip(axs_ts, ts_pairs):
        r, g = metric_arrays[key]
        if r.size == len(t_r):
            ax.plot(t_r, r, color=RAW_COLOR, lw=1.1, label="Raw")
        if g.size == len(t_g):
            ax.plot(t_g, g, color=GEN_COLOR, lw=1.1, alpha=0.9, label="Generated")
        ax.axvline(takeover_time_s, color="0.3", ls=":", lw=1.0)
        ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)
    axs_ts[0].set_title(title_suffix)
    axs_ts[-1].set_xlabel("Time (s)")
    fig_ts.tight_layout()
    ts_png = out_dir / driver / "consistency_timeseries.png"
    ts_png.parent.mkdir(parents=True, exist_ok=True)
    fig_ts.savefig(ts_png, bbox_inches="tight")
    plt.close(fig_ts)

    fig_risk, axs_risk = plt.subplots(2, 1, figsize=(12, 5.5), dpi=120, sharex=True)
    for ax, key, ylab in zip(axs_risk, ("thw", "ttc"), ("THW (s)", "TTC (s)")):
        r, g = metric_arrays[key]
        if r.size == len(t_r):
            ax.plot(t_r, r, color=RAW_COLOR, lw=1.0, label="Raw")
        if g.size == len(t_g):
            ax.plot(t_g, g, color=GEN_COLOR, lw=1.0, label="Generated")
        ax.axvline(takeover_time_s, color="0.3", ls=":", lw=1.0)
        ax.set_ylabel(ylab)
        ax.grid(True, ls="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)
    axs_risk[-1].set_xlabel("Time (s)")
    fig_risk.suptitle("{} risk metrics".format(driver), fontsize=10)
    fig_risk.tight_layout()
    risk_png = out_dir / driver / "consistency_risk_timeseries.png"
    fig_risk.savefig(risk_png, bbox_inches="tight")
    plt.close(fig_risk)

    hist_metrics = ("speed", "accel", "headway", "thw", "ttc", "inv_thw", "inv_ttc", "jerk")
    hist_labels = {
        "speed": "Speed (m/s)",
        "accel": "Accel (m/s²)",
        "headway": "Headway (m)",
        "thw": "THW (s)",
        "ttc": "TTC (s)",
        "inv_thw": "1/THW (1/s)",
        "inv_ttc": "1/TTC (1/s)",
        "jerk": "Jerk (m/s³)",
    }
    fig_hist, axs_hist = plt.subplots(4, 2, figsize=(12, 12), dpi=120)
    for ax, key in zip(axs_hist.ravel(), hist_metrics):
        r, g = metric_arrays[key]
        r = r[np.isfinite(r)]
        g = g[np.isfinite(g)]
        if r.size:
            ax.hist(r, bins=40, density=True, alpha=0.45, color=RAW_COLOR, label="Raw")
        if g.size:
            ax.hist(g, bins=40, density=True, alpha=0.45, color=GEN_COLOR, label="Gen")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel(hist_labels[key], fontsize=8)
        ax.grid(True, ls="--", alpha=0.25)
        if key == "speed":
            ax.legend(fontsize=7)
    fig_hist.suptitle("{} distributions".format(driver), fontsize=10)
    fig_hist.tight_layout()
    hist_png = out_dir / driver / "consistency_distributions.png"
    fig_hist.savefig(hist_png, bbox_inches="tight")
    plt.close(fig_hist)


def _plot_fleet(summary_df: pd.DataFrame, out_png: Path) -> None:
    if summary_df.empty:
        return
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(11, 9), dpi=120)
    d = summary_df.copy()

    ax = axs[0, 0]
    ax.scatter(d["raw_headway_mean"], d["gen_headway_mean"], c=np.where(d["all_ok"], PASS_COLOR, FAIL_COLOR))
    lim = [
        min(d["raw_headway_mean"].min(), d["gen_headway_mean"].min()) * 0.95,
        max(d["raw_headway_mean"].max(), d["gen_headway_mean"].max()) * 1.05,
    ]
    ax.plot(lim, lim, "k--", lw=1.0, alpha=0.5)
    ax.set_xlabel("Raw headway mean (m)")
    ax.set_ylabel("Generated headway mean (m)")
    ax.set_title("Headway mean")
    ax.grid(True, ls="--", alpha=0.3)

    ax = axs[0, 1]
    ax.bar(d["driver_id"], d["gap_mean_ratio"] - 1.0, color=np.where(d["gap_mean_ok"], PASS_COLOR, FAIL_COLOR))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("Gap mean ratio - 1")
    ax.set_title("Headway mean drift")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    ax = axs[1, 0]
    ax.bar(d["driver_id"], d["geom_gap_mae_m"], color=np.where(d["geometry_ok"], PASS_COLOR, FAIL_COLOR))
    ax.set_ylabel("Geometry gap MAE (m)")
    ax.set_title("Gap vs position consistency")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    ax = axs[1, 1]
    ax.bar(d["driver_id"], d["thw_abnormal_frac"], color=RAW_COLOR, alpha=0.75, label="THW abnormal frac")
    ax.bar(d["driver_id"], d["ttc_abnormal_frac"], color=GEN_COLOR, alpha=0.75, label="TTC abnormal frac")
    ax.set_ylabel("Fraction")
    ax.set_title("Risk metric abnormal fraction (TTC informational)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", ls="--", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate generated takeover trajectory consistency vs raw.")
    ap.add_argument(
        "--generated_dir",
        type=Path,
        default=Path("following/outputs/residual_gru_takeover_20s_yaw_shrink_controls"),
    )
    ap.add_argument(
        "--generation_summary",
        type=Path,
        default=Path("following/outputs/residual_gru_takeover_20s/generation_summary.csv"),
        help="CSV with driver_id,source_csv mapping.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("following/outputs/pictures/takeover_consistency_yaw_shrink_controls"),
    )
    ap.add_argument("--min_time_s", type=float, default=20.0, help="Analyze rows with t >= min_time_s.")
    ap.add_argument("--max_time_s", type=float, default=-1.0, help="Optional upper time bound; negative disables.")
    ap.add_argument("--takeover_time_s", type=float, default=20.0)
    ap.add_argument("--gap_mean_ratio_tol", type=float, default=0.30)
    ap.add_argument("--gap_std_ratio_tol", type=float, default=0.65)
    ap.add_argument("--accel_min_mps2", type=float, default=-8.0)
    ap.add_argument("--accel_max_mps2", type=float, default=6.0)
    ap.add_argument("--risk_min_s", type=float, default=0.3)
    ap.add_argument("--risk_max_s", type=float, default=120.0)
    ap.add_argument("--risk_abnormal_frac_tol", type=float, default=0.05)
    ap.add_argument("--geom_gap_mae_tol_m", type=float, default=0.15)
    ap.add_argument("--rel_speed_mae_tol_mps", type=float, default=0.25)
    ap.add_argument("--drivers", type=str, default="")
    ap.add_argument("--csv_only", action="store_true", help="Skip PNG generation; write CSV/JSON only.")
    args = ap.parse_args()

    gen_dir = args.generated_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    max_time_s = None if args.max_time_s < 0 else float(args.max_time_s)

    sources = _load_generation_sources(args.generation_summary.expanduser().resolve())
    if not sources and (gen_dir / "generation_summary.csv").is_file():
        sources = _load_generation_sources(gen_dir / "generation_summary.csv")

    if args.drivers.strip():
        drivers = [x.strip() for x in args.drivers.split(",") if x.strip()]
    else:
        drivers = sorted(
            [p.name for p in gen_dir.iterdir() if p.is_dir() and p.name.startswith("T")],
            key=_driver_sort_key,
        )

    metric_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for driver in drivers:
        gen_path = gen_dir / driver / "driving_data.csv"
        raw_path = sources.get(driver)
        if raw_path is None or not raw_path.is_file():
            print("[SKIP] {} missing raw source".format(driver))
            continue
        if not gen_path.is_file():
            print("[SKIP] {} missing generated CSV".format(driver))
            continue

        df_raw = _load_df(raw_path)
        df_gen = _load_df(gen_path)
        if df_raw is None or df_gen is None:
            print("[SKIP] {} failed to load".format(driver))
            continue

        raw_w = _slice_window(df_raw, args.min_time_s, max_time_s)
        gen_w = _slice_window(df_gen, args.min_time_s, max_time_s)
        if raw_w.empty or gen_w.empty:
            print("[SKIP] {} empty analysis window".format(driver))
            continue

        metric_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for key, cols, _, mask_risk in METRIC_SPECS:
            r = _series_from_df(raw_w, cols, mask_risk)
            g = _series_from_df(gen_w, cols, mask_risk)
            metric_arrays[key] = (r, g)
            sr, sg = _stats(r), _stats(g)
            metric_rows.append(dict(
                driver_id=driver,
                metric=key,
                raw_n=sr["n"], raw_mean=sr["mean"], raw_std=sr["std"],
                raw_p05=sr["p05"], raw_p50=sr["p50"], raw_p95=sr["p95"],
                gen_n=sg["n"], gen_mean=sg["mean"], gen_std=sg["std"],
                gen_p05=sg["p05"], gen_p50=sg["p50"], gen_p95=sg["p95"],
                mean_diff=sg["mean"] - sr["mean"] if sr["mean"] == sr["mean"] and sg["mean"] == sg["mean"] else float("nan"),
                std_ratio=sg["std"] / sr["std"] if sr["std"] == sr["std"] and sr["std"] > 1e-6 else float("nan"),
                ks_distance=_ks_distance(r, g),
                min_time_s=args.min_time_s,
            ))

        raw_dict = {k: metric_arrays[k][0] for k, _, _, _ in METRIC_SPECS}
        gen_dict = {k: metric_arrays[k][1] for k, _, _, _ in METRIC_SPECS}
        geom_err = _geometry_gap_error(gen_w)
        rel_internal_raw = _relative_speed_internal_mae(raw_w)
        rel_internal_gen = _relative_speed_internal_mae(gen_w)
        flags = _check_flags(
            raw_dict, gen_dict, geom_err, rel_internal_raw, rel_internal_gen, args,
        )

        summary_rows.append(dict(
            driver_id=driver,
            raw_csv=str(raw_path),
            gen_csv=str(gen_path),
            n_raw=int(len(raw_w)),
            n_gen=int(len(gen_w)),
            raw_headway_mean=_stats(raw_dict["headway"])["mean"],
            gen_headway_mean=_stats(gen_dict["headway"])["mean"],
            raw_headway_std=_stats(raw_dict["headway"])["std"],
            gen_headway_std=_stats(gen_dict["headway"])["std"],
            **flags,
        ))

        driver_out = out_dir / driver
        if not args.csv_only:
            try:
                _plot_driver(driver, raw_w, gen_w, metric_arrays, flags, driver_out, args.takeover_time_s)
                print("[OK] {} -> {}/consistency_*.png".format(driver, driver_out / driver))
            except ImportError as exc:
                print("[WARN] {} plotting skipped ({})".format(driver, exc))
        else:
            print("[OK] {} metrics computed (csv_only)".format(driver))

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(out_dir / "metric_distribution_comparison.csv", index=False)
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(out_dir / "consistency_summary.csv", index=False)
        if not args.csv_only:
            try:
                _plot_fleet(sdf, out_dir / "fleet_consistency_summary.png")
            except ImportError as exc:
                print("[WARN] fleet plot skipped ({})".format(exc))
        n_pass = int(sdf["all_ok"].sum())
        print("\n[DONE] {} drivers | PASS {}/{}".format(len(sdf), n_pass, len(sdf)))
        print("Summary: {}".format(out_dir / "consistency_summary.csv"))
        with open(out_dir / "consistency_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                dict(
                    generated_dir=str(gen_dir),
                    min_time_s=args.min_time_s,
                    n_drivers=len(sdf),
                    n_pass=n_pass,
                    thresholds=dict(
                        gap_mean_ratio_tol=args.gap_mean_ratio_tol,
                        gap_std_ratio_tol=args.gap_std_ratio_tol,
                        accel_range=[args.accel_min_mps2, args.accel_max_mps2],
                        thw_abnormal_frac_tol=args.risk_abnormal_frac_tol,
                        ttc_risk_checked=False,
                        geom_gap_mae_tol_m=args.geom_gap_mae_tol_m,
                        rel_speed_mae_tol_mps=args.rel_speed_mae_tol_mps,
                    ),
                ),
                f,
                indent=2,
            )
    else:
        print("[WARN] No drivers processed.")


if __name__ == "__main__":
    main()
