# -*- coding: utf-8 -*-
"""
Overlay comparison: closed-loop residual GRU takeover vs original driver log.

Plots per-driver figures with closed-loop and ground-truth series on the same
axes (position, speed, acceleration, distance headway), plus an extra row
showing the IDM gain/residual diagnostics (``gru_alpha`` / ``gru_delta_a`` /
``a_idm_base``) so you can tell how much of each acceleration sample came from
the IDM baseline vs the learned residual.

Usage::

    python3 following/scripts/visualize_takeover_vs_raw.py \
        --takeover_dir following/outputs/residual_gru_takeover_20s \
        --takeover_time_s 20.0

By default the raw CSV for each driver is read from
``<takeover_dir>/generation_summary.csv`` (``source_csv`` column), so the plot
compares each generated trajectory against the exact calibrated session used to
generate it. Use ``--raw_selection index`` to fall back to the old
``--session_index`` lookup.

Output PNGs go to
``following/outputs/pictures/residual_gru_takeover_vs_raw/<T*>/driving_data.png``.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _raw_path_for(driver: str, raw_dir: Path, session_index: int) -> Optional[Path]:
    sessions = sorted(glob.glob(str(raw_dir / driver / "行车" / "*")))
    if not sessions:
        return None
    idx = session_index if session_index >= 0 else len(sessions) + session_index
    if idx < 0 or idx >= len(sessions):
        return None
    return Path(sessions[idx]) / "driving_data.csv"


def _load_generation_sources(takeover_dir: Path) -> Dict[str, Path]:
    summary_csv = takeover_dir / "generation_summary.csv"
    out: Dict[str, Path] = {}
    if not summary_csv.is_file():
        return out
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver = str(row.get("driver_id", "")).strip()
            source = str(row.get("source_csv", "")).strip()
            if driver and source:
                out[driver] = Path(source)
    return out


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    tcol = "sim_time_s" if "sim_time_s" in df.columns else "timestamp"
    df["__t"] = df[tcol].astype(float)
    return df


def _align_frames(df_raw: pd.DataFrame, df_take: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crop both series to the common sim-time window."""
    t_lo = max(df_raw["__t"].min(), df_take["__t"].min())
    t_hi = min(df_raw["__t"].max(), df_take["__t"].max())
    raw = df_raw[(df_raw["__t"] >= t_lo) & (df_raw["__t"] <= t_hi)].reset_index(drop=True)
    take = df_take[(df_take["__t"] >= t_lo) & (df_take["__t"] <= t_hi)].reset_index(drop=True)
    return raw, take


def _metrics_after(raw: pd.DataFrame, take: pd.DataFrame, t_start: float) -> dict:
    """RMSE / bias after the takeover point, evaluated on overlapping rows."""
    def clip(df):
        return df[df["__t"] >= t_start].reset_index(drop=True)
    r, c = clip(raw), clip(take)
    n = min(len(r), len(c))
    if n == 0:
        return {}
    r, c = r.iloc[:n], c.iloc[:n]

    def rmse(a, b):
        diff = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
        return float(np.sqrt(np.mean(diff ** 2)))

    def mean_diff(a, b):
        return float(np.mean(c[b].to_numpy(dtype=float) - r[a].to_numpy(dtype=float)))

    return dict(
        n=n,
        rmse_v=rmse(r["ego_speed"], c["ego_speed"]),
        rmse_a=rmse(r["ego_acceleration"], c["ego_acceleration"]),
        rmse_gap=rmse(r["distance_headway"], c["distance_headway"]),
        mean_a_raw=float(r["ego_acceleration"].mean()),
        mean_a_take=float(c["ego_acceleration"].mean()),
        std_a_raw=float(r["ego_acceleration"].std()),
        std_a_take=float(c["ego_acceleration"].std()),
        mean_gap_raw=float(r["distance_headway"].mean()),
        mean_gap_take=float(c["distance_headway"].mean()),
    )


def _plot_overlay(driver: str,
                  raw: pd.DataFrame,
                  take: pd.DataFrame,
                  out_png: Path,
                  takeover_time_s: float,
                  metrics: dict) -> None:
    has_alpha = "gru_alpha" in take.columns and take["gru_alpha"].notna().any()
    n_rows = 5 if has_alpha else 4
    fig, axs = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), dpi=120, sharex=True)

    t_raw = raw["__t"].to_numpy()
    t_tk = take["__t"].to_numpy()

    # --- 0. Position
    axs[0].plot(t_raw, raw["ego_pos_x"], color="#1f77b4", lw=1.3, label="Ego (raw)")
    axs[0].plot(t_tk, take["ego_pos_x"], color="#1f77b4", lw=1.3, ls="--", alpha=0.75,
                label="Ego (GRU+IDM)")
    axs[0].plot(t_raw, raw["lead_pos_x"], color="#ff7f0e", lw=1.3, label="Lead")
    axs[0].set_ylabel("Position X (m)")
    axs[0].set_title("{}   takeover @ t={:.1f}s   (raw vs closed-loop GRU+IDM)".format(
        driver, takeover_time_s))
    axs[0].legend(loc="upper left", fontsize=9)
    axs[0].grid(True, ls="--", alpha=0.5)

    # --- 1. Speed
    axs[1].plot(t_raw, raw["ego_speed"], color="#1f77b4", lw=1.3, label="Ego (raw)")
    axs[1].plot(t_tk, take["ego_speed"], color="#d62728", lw=1.3, label="Ego (closed-loop)")
    axs[1].plot(t_raw, raw["lead_speed"], color="#ff7f0e", lw=1.1, alpha=0.7, label="Lead")
    axs[1].set_ylabel("Speed (m/s)")
    axs[1].legend(loc="upper right", fontsize=9)
    axs[1].grid(True, ls="--", alpha=0.5)
    if metrics:
        axs[1].text(0.01, 0.95, "RMSE v = {:.2f} m/s".format(metrics["rmse_v"]),
                    transform=axs[1].transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # --- 2. Acceleration
    axs[2].plot(t_raw, raw["ego_acceleration"], color="#1f77b4", lw=1.1, label="Ego a (raw)")
    axs[2].plot(t_tk, take["ego_acceleration"], color="#d62728", lw=1.1,
                label="Ego a (closed-loop)")
    axs[2].set_ylabel("Accel (m/s²)")
    axs[2].legend(loc="upper right", fontsize=9)
    axs[2].grid(True, ls="--", alpha=0.5)
    if metrics:
        axs[2].text(0.01, 0.95,
                    "raw σa={:.2f}  closed σa={:.2f}".format(
                        metrics["std_a_raw"], metrics["std_a_take"]),
                    transform=axs[2].transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # --- 3. Distance headway
    axs[3].plot(t_raw, raw["distance_headway"], color="#1f77b4", lw=1.3, label="Gap (raw)")
    axs[3].plot(t_tk, take["distance_headway"], color="#d62728", lw=1.3,
                label="Gap (closed-loop)")
    axs[3].axhline(0, color="black", ls="--", lw=1, alpha=0.5)
    axs[3].set_ylabel("Gap (m)")
    axs[3].legend(loc="upper right", fontsize=9)
    axs[3].grid(True, ls="--", alpha=0.5)
    if metrics:
        axs[3].text(0.01, 0.95,
                    "raw gap={:.1f}m  closed gap={:.1f}m  RMSE={:.1f}m".format(
                        metrics["mean_gap_raw"], metrics["mean_gap_take"],
                        metrics["rmse_gap"]),
                    transform=axs[3].transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    # --- 4. GRU internals (only when present)
    if has_alpha:
        ax = axs[4]
        ax2 = ax.twinx()
        # alpha & a_idm on left axis; delta on right
        mask = take["gru_alpha"].notna()
        ax.plot(t_tk[mask], take.loc[mask, "gru_alpha"], color="#2ca02c",
                lw=1.1, label="α (IDM gain)")
        if "a_idm_base" in take.columns:
            ax.plot(t_tk[mask], take.loc[mask, "a_idm_base"], color="#8c564b",
                    lw=1.0, alpha=0.7, label="a_IDM_base")
        ax.set_ylabel("α  /  a_IDM")
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)
        if "gru_delta_a" in take.columns:
            ax2.plot(t_tk[mask], take.loc[mask, "gru_delta_a"], color="#9467bd",
                     lw=1.0, alpha=0.85, label="Δa (residual)")
        if "gap_anchor_a" in take.columns:
            ax2.plot(t_tk[mask], take.loc[mask, "gap_anchor_a"], color="#ff7f0e",
                     lw=1.0, ls="--", alpha=0.85, label="gap anchor a")
        if "gru_delta_a" in take.columns or "gap_anchor_a" in take.columns:
            ax2.set_ylabel("Δa (m/s²)", color="#9467bd")
            ax2.legend(loc="upper right", fontsize=9)

    # Vertical line at takeover
    for ax in axs:
        ax.axvline(takeover_time_s, color="black", ls=":", lw=1.2, alpha=0.7)
        ax.text(takeover_time_s, ax.get_ylim()[1], " takeover",
                ha="left", va="top", fontsize=8, color="black")

    axs[-1].set_xlabel("sim_time (s)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--takeover_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/residual_gru_takeover_20s")
    ap.add_argument("--raw_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/following_calibrated")
    ap.add_argument("--out_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/pictures/residual_gru_takeover_vs_raw")
    ap.add_argument("--session_index", type=int, default=-1,
                    help="0-based index of the raw session to compare against. "
                         "Only used with --raw_selection index. Default -1 = last session.")
    ap.add_argument("--raw_selection", type=str, default="generation_summary",
                    choices=["generation_summary", "index"],
                    help="How to choose the raw CSV. Default uses takeover_dir/generation_summary.csv.")
    ap.add_argument("--takeover_time_s", type=float, default=20.0)
    ap.add_argument("--drivers", type=str, default="")
    args = ap.parse_args()

    take_dir = Path(args.takeover_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    generation_sources = _load_generation_sources(take_dir) if args.raw_selection == "generation_summary" else {}
    if args.raw_selection == "generation_summary":
        if generation_sources:
            print("[INFO] loaded raw source mapping from {}".format(take_dir / "generation_summary.csv"))
        else:
            print("[WARN] generation_summary.csv not found or empty; falling back to --session_index lookup.")

    # enumerate drivers from takeover_dir
    drivers = []
    if args.drivers:
        drivers = [d.strip() for d in args.drivers.split(",") if d.strip()]
    else:
        drivers = sorted([p.name for p in take_dir.iterdir() if p.is_dir() and p.name.startswith("T")],
                         key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999)

    summary_rows = []
    for d in drivers:
        take_csv = take_dir / d / "driving_data.csv"
        raw_csv = generation_sources.get(d)
        raw_source = "generation_summary"
        if raw_csv is None or not raw_csv.is_file():
            raw_csv = _raw_path_for(d, raw_dir, args.session_index)
            raw_source = "index:{}".format(args.session_index)
        if raw_csv is None or not raw_csv.is_file():
            print("[SKIP] {}: raw CSV not found (session_index={})".format(d, args.session_index))
            continue
        if not take_csv.is_file():
            print("[SKIP] {}: no takeover CSV at {}".format(d, take_csv))
            continue

        raw = _load_csv(raw_csv)
        take = _load_csv(take_csv)
        if raw is None or take is None:
            continue
        raw, take = _align_frames(raw, take)
        metrics = _metrics_after(raw, take, args.takeover_time_s)

        out_png = out_dir / d / "driving_data.png"
        _plot_overlay(d, raw, take, out_png, args.takeover_time_s, metrics)
        print("[OK] {} -> {}   (RMSE v={:.2f} a={:.2f} gap={:.2f})".format(
            d, out_png,
            metrics.get("rmse_v", float("nan")),
            metrics.get("rmse_a", float("nan")),
            metrics.get("rmse_gap", float("nan"))))
        summary_rows.append(dict(
            driver=d,
            raw_csv=str(raw_csv),
            raw_source=raw_source,
            take_csv=str(take_csv),
            out_png=str(out_png),
            **metrics,
        ))

    if summary_rows:
        import csv as _csv
        sum_fp = out_dir / "overlay_summary.csv"
        sum_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(sum_fp, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print("\n[DONE] {} plots. Summary: {}".format(len(summary_rows), sum_fp))


if __name__ == "__main__":
    main()
