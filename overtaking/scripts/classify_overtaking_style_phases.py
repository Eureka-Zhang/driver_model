# -*- coding: utf-8 -*-
"""
Classify per-driver overtaking style into 3 ordered labels:
  conservative / neutral / aggressive

This script uses four segmented overtaking phases:
  1) following
  2) lane_change
  3) left_overtake
  4) return

Method
------
For each overtaking maneuver, extract phase-aware behavioural features, then
aggregate them per driver. Drivers are projected to three interpretable axes:

  S: Safety margin / conservativeness
     - larger initial TTC / initial gap / following THW => more conservative

  L: Lateral intensity
     - larger lateral acceleration / lateral jerk / shorter lane-change duration
       => more aggressive

  P: Passing assertiveness
     - larger passing speed advantage / shorter total duration => more aggressive

Composite score:

    style_score = w_L * L + w_P * P - w_S * S

Drivers are sorted by style_score and split into terciles:
  bottom third -> conservative
  middle third -> neutral
  top third    -> aggressive

Outputs
-------
<out_dir>/overtaking_style_labels.json
<out_dir>/overtaking_style_prototypes.json
<out_dir>/overtaking_style_features.csv
<out_dir>/overtaking_style_heatmap.png
<out_dir>/overtaking_style_scatter.png

Usage
-----
python3 overtaking/scripts/classify_overtaking_style_phases.py
python3 overtaking/scripts/classify_overtaking_style_phases.py --plot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


PHASE_NAMES = ("following", "lane_change", "left_overtake", "return")

AXIS_S_FEATURES = ["ttc_init", "dist_init", "thw_follow_median"]
AXIS_L_FEATURES = ["lat_acc_peak", "lat_jerk_peak", "lc_duration_inv"]
AXIS_P_FEATURES = ["pass_speed_adv", "total_duration_inv", "return_speed_peak"]
ALL_SCORE_FEATURES = AXIS_S_FEATURES + AXIS_L_FEATURES + AXIS_P_FEATURES


def _parse_float(v: object) -> Optional[float]:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    arr = sorted(float(v) for v in vals)
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * (q / 100.0)
    i = int(math.floor(k))
    j = min(i + 1, len(arr) - 1)
    a = k - i
    return arr[i] * (1.0 - a) + arr[j] * a


def _median(vals: List[float]) -> float:
    return _percentile(vals, 50.0)


def _driver_id_from_path(path: Path) -> str:
    p = str(path).replace("\\", "/")
    m = re.search(r"/(T-?\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _driver_sort_key(did: str) -> Tuple[int, str]:
    m = re.match(r"^T(\d+)$", did)
    if m:
        return (int(m.group(1)), did)
    return (10**9, did)


def _discover_maneuvers(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    phase_map: Dict[str, Dict[str, Path]] = {}
    for path in data_dir.rglob("*.csv"):
        if path.name == "phase_segments_summary.csv":
            continue
        m = re.match(r"^(.+?)__phase_(\d+)_(following|lane_change|left_overtake|return)\.csv$", path.name)
        if not m:
            continue
        stem = str(path.parent / m.group(1))
        phase_name = m.group(3)
        phase_map.setdefault(stem, {})[phase_name] = path
    return phase_map


def _read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _series_from_rows(rows: List[dict], key: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        v = _parse_float(r.get(key))
        if v is not None:
            vals.append(v)
    return vals


def _time_range(rows: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    ts = _series_from_rows(rows, "timestamp")
    if not ts:
        return None, None
    return ts[0], ts[-1]


def _central_derivative(times: List[float], vals: List[float]) -> List[float]:
    n = min(len(times), len(vals))
    if n == 0:
        return []
    out = [0.0] * n
    if n >= 3:
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if abs(dt) > 1e-9:
                out[i] = (vals[i + 1] - vals[i - 1]) / dt
    if n >= 2:
        dt0 = times[1] - times[0]
        if abs(dt0) > 1e-9:
            out[0] = (vals[1] - vals[0]) / dt0
        dt1 = times[-1] - times[-2]
        if abs(dt1) > 1e-9:
            out[-1] = (vals[-1] - vals[-2]) / dt1
    return out


def _extract_maneuver_features(phase_paths: Dict[str, Path]) -> Optional[dict]:
    if any(name not in phase_paths for name in PHASE_NAMES):
        return None

    rows = {name: _read_csv_rows(phase_paths[name]) for name in PHASE_NAMES}
    if any(len(rows[name]) < 2 for name in PHASE_NAMES):
        return None

    follow = rows["following"]
    lane_change = rows["lane_change"]
    left_ov = rows["left_overtake"]
    ret = rows["return"]

    # --- initial safety margin at lane change start ---
    first_lc = lane_change[0]
    ttc_init = _parse_float(first_lc.get("ttc"))
    if ttc_init is not None and abs(ttc_init - 999.0) < 1e-3:
        ttc_init = None
    if ttc_init is None:
        gap0 = _parse_float(first_lc.get("distance_headway"))
        ego0 = _parse_float(first_lc.get("ego_speed"))
        lead0 = _parse_float(first_lc.get("lead_speed"))
        if gap0 is not None and ego0 is not None and lead0 is not None:
            rel = ego0 - lead0
            if rel > 1e-2:
                ttc_init = gap0 / rel
    dist_init = _parse_float(first_lc.get("distance_headway"))

    # --- following phase closeness ---
    follow_thw = [v for v in _series_from_rows(follow, "time_headway") if 0.0 < v < 100.0 and abs(v - 999.0) > 1e-3]
    thw_follow_median = _median(follow_thw) if follow_thw else float("nan")

    # --- lateral intensity over lane change + return ---
    lat_rows = lane_change + ret
    lat_t = _series_from_rows(lat_rows, "timestamp")
    lat_y = _series_from_rows(lat_rows, "ego_pos_y")
    if len(lat_t) != len(lat_y) or len(lat_t) < 3:
        return None
    lat_v = _central_derivative(lat_t, lat_y)
    lat_a = _central_derivative(lat_t, lat_v)
    lat_j = _central_derivative(lat_t, lat_a)
    lat_acc_peak = max((abs(x) for x in lat_a), default=float("nan"))
    lat_jerk_peak = max((abs(x) for x in lat_j), default=float("nan"))

    # --- durations ---
    t0_lc, t1_lc = _time_range(lane_change)
    t0_ret, t1_ret = _time_range(ret)
    t0_left, t1_left = _time_range(left_ov)
    if None in (t0_lc, t1_lc, t0_ret, t1_ret, t0_left, t1_left):
        return None
    lc_duration = float(t1_lc - t0_lc)
    total_duration = float(t1_ret - t0_lc)

    # --- passing assertiveness ---
    pass_adv = []
    for r in left_ov:
        ego = _parse_float(r.get("ego_speed"))
        lead = _parse_float(r.get("lead_speed"))
        if ego is None or lead is None:
            continue
        pass_adv.append(ego - lead)
    pass_speed_adv = _mean(pass_adv) if pass_adv else float("nan")
    return_speed_vals = _series_from_rows(ret, "ego_speed")
    return_speed_peak = max(return_speed_vals) if return_speed_vals else float("nan")

    return {
        "ttc_init": ttc_init,
        "dist_init": dist_init,
        "thw_follow_median": thw_follow_median,
        "lat_acc_peak": lat_acc_peak,
        "lat_jerk_peak": lat_jerk_peak,
        "lc_duration": lc_duration,
        "lc_duration_inv": (1.0 / lc_duration) if lc_duration and lc_duration > 1e-6 else float("nan"),
        "pass_speed_adv": pass_speed_adv,
        "total_duration": total_duration,
        "total_duration_inv": (1.0 / total_duration) if total_duration and total_duration > 1e-6 else float("nan"),
        "return_speed_peak": return_speed_peak,
    }


def _aggregate_driver_features(records: List[dict]) -> Optional[dict]:
    if not records:
        return None
    out = {"n_segments_valid": len(records)}
    keys = list(records[0].keys())
    for k in keys:
        vals = [r[k] for r in records if r.get(k) is not None and math.isfinite(r[k])]
        if not vals:
            out[k] = float("nan")
        else:
            out[k] = _median(vals)
    return out


def _zscore_table(drivers: List[str], features: Dict[str, dict], keys: List[str]) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    arr = []
    for d in drivers:
        row = []
        for k in keys:
            row.append(float(features[d][k]))
        arr.append(row)
    x = np.asarray(arr, dtype=float)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    z = (x - means) / stds
    return z, {k: float(means[i]) for i, k in enumerate(keys)}, {k: float(stds[i]) for i, k in enumerate(keys)}


def _compute_style_scores(drivers: List[str], features: Dict[str, dict], w_S: float = 1.0, w_L: float = 1.0, w_P: float = 1.0):
    z_all, mean_all, std_all = _zscore_table(drivers, features, ALL_SCORE_FEATURES)
    idx = {k: i for i, k in enumerate(ALL_SCORE_FEATURES)}
    results = []
    for r_i, d in enumerate(drivers):
        S = float(np.mean([z_all[r_i, idx[k]] for k in AXIS_S_FEATURES]))
        L = float(np.mean([z_all[r_i, idx[k]] for k in AXIS_L_FEATURES]))
        P = float(np.mean([z_all[r_i, idx[k]] for k in AXIS_P_FEATURES]))
        score = w_L * L + w_P * P - w_S * S
        results.append({"driver": d, "score": score, "S": S, "L": L, "P": P})
    results.sort(key=lambda r: r["score"])
    return results, z_all, mean_all, std_all


def _assign_labels(sorted_results: List[dict]) -> Dict[str, str]:
    n = len(sorted_results)
    n_cons = n // 3
    n_aggr = n // 3
    n_neut = n - n_cons - n_aggr
    labels: Dict[str, str] = {}
    for i, r in enumerate(sorted_results):
        if i < n_cons:
            labels[r["driver"]] = "conservative"
        elif i < n_cons + n_neut:
            labels[r["driver"]] = "neutral"
        else:
            labels[r["driver"]] = "aggressive"
    return labels


def _find_prototypes(sorted_results: List[dict], labels: Dict[str, str]) -> Dict[str, str]:
    groups: Dict[str, List[dict]] = {}
    for r in sorted_results:
        groups.setdefault(labels[r["driver"]], []).append(r)
    out: Dict[str, str] = {}
    for lbl, members in groups.items():
        if lbl == "conservative":
            best = max(members, key=lambda m: m["S"] - m["L"] - m["P"])
        elif lbl == "aggressive":
            best = max(members, key=lambda m: m["L"] + m["P"] - m["S"])
        else:
            best = min(members, key=lambda m: abs(m["S"]) + abs(m["L"]) + abs(m["P"]))
        out[lbl] = best["driver"]
    return out


def _plot_heatmap(sorted_results: List[dict], z_all: np.ndarray, driver_order: List[str], labels: Dict[str, str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    order_map = {d: i for i, d in enumerate(driver_order)}
    order = [order_map[r["driver"]] for r in sorted_results]
    z_sorted = z_all[order]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(z_sorted, aspect="auto", cmap="RdYlGn_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels([
        "{} {} ({:.2f})".format({"conservative": "●", "neutral": "◆", "aggressive": "▲"}[labels[r["driver"]]], r["driver"], r["score"])
        for r in sorted_results
    ], fontsize=9)
    ax.set_xticks(range(len(ALL_SCORE_FEATURES)))
    ax.set_xticklabels(ALL_SCORE_FEATURES, rotation=45, ha="right", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("z-score")
    ax.set_title("Overtaking Style Features (sorted by aggressiveness score)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(sorted_results: List[dict], labels: Dict[str, str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"conservative": "#2ca02c", "neutral": "#1f77b4", "aggressive": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for r in sorted_results:
        lbl = labels[r["driver"]]
        ax.scatter(r["S"], r["L"] + r["P"], color=colors[lbl], s=80, zorder=3)
        ax.annotate(r["driver"], (r["S"], r["L"] + r["P"]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("S (Safety margin, z-score) ← conservative")
    ax.set_ylabel("L + P (Lateral intensity + Passing assertiveness) → aggressive")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_title("Overtaking Style: Safety vs Intensity+Passing")
    ax.legend(handles=[
        Patch(facecolor=colors["conservative"], label="Conservative"),
        Patch(facecolor=colors["neutral"], label="Neutral"),
        Patch(facecolor=colors["aggressive"], label="Aggressive"),
    ], loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify overtaking style into conservative/neutral/aggressive using 4 phases.")
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/overtaking/outputs/overtaking_phase_segments_all_p1p3", help="Root directory with phase_01~04 CSVs.")
    ap.add_argument("--out_dir", type=str, default="/home/zwx/driver_model/overtaking/outputs/overtaking_style_phase_score", help="Output directory.")
    ap.add_argument("--w_S", type=float, default=1.0, help="Weight for Safety axis.")
    ap.add_argument("--w_L", type=float, default=1.0, help="Weight for Lateral intensity axis.")
    ap.add_argument("--w_P", type=float, default=1.0, help="Weight for Passing axis.")
    ap.add_argument("--plot", action="store_true", help="Generate heatmap and scatter plots.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit("[ERR] data_dir not found: {}".format(data_dir))

    phase_map = _discover_maneuvers(data_dir)
    by_driver: Dict[str, List[dict]] = {}
    n_total = 0
    n_valid = 0
    for stem, phases in sorted(phase_map.items()):
        n_total += 1
        feat = _extract_maneuver_features(phases)
        if feat is None:
            continue
        did = _driver_id_from_path(Path(stem))
        by_driver.setdefault(did, []).append(feat)
        n_valid += 1

    drivers = sorted(by_driver.keys(), key=_driver_sort_key)
    if len(drivers) < 3:
        raise SystemExit("[ERR] need at least 3 drivers with valid overtaking maneuvers, got {}".format(len(drivers)))

    features: Dict[str, dict] = {}
    for d in drivers:
        agg = _aggregate_driver_features(by_driver[d])
        if agg is not None:
            features[d] = agg

    drivers = [d for d in drivers if d in features]
    if len(drivers) < 3:
        raise SystemExit("[ERR] need at least 3 drivers after aggregation, got {}".format(len(drivers)))

    sorted_results, z_all, mean_all, std_all = _compute_style_scores(drivers, features, w_S=args.w_S, w_L=args.w_L, w_P=args.w_P)
    labels = _assign_labels(sorted_results)
    prototypes = _find_prototypes(sorted_results, labels)

    print("[data] maneuvers total={}, valid={}".format(n_total, n_valid))
    print("[data] drivers used={}".format(len(drivers)))
    print("\n{:<5s} {:>7s} {:>6s} {:>6s} {:>6s}  {:<14s}".format("Drv", "Score", "S", "L", "P", "Label"))
    print("-" * 52)
    for r in sorted_results:
        print("{:<5s} {:+7.3f} {:+6.3f} {:+6.3f} {:+6.3f}  {:<14s}".format(
            r["driver"], r["score"], r["S"], r["L"], r["P"], labels[r["driver"]]))
    print("\nPrototypes: {}".format(prototypes))

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_fp = out_dir / "overtaking_style_labels.json"
    with labels_fp.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    proto_fp = out_dir / "overtaking_style_prototypes.json"
    with proto_fp.open("w", encoding="utf-8") as f:
        json.dump({
            "prototypes": prototypes,
            "sorted_scores": [
                dict(driver=r["driver"], score=r["score"], S=r["S"], L=r["L"], P=r["P"], label=labels[r["driver"]])
                for r in sorted_results
            ],
        }, f, indent=2, ensure_ascii=False)

    feat_fp = out_dir / "overtaking_style_features.csv"
    feat_keys = ["n_segments_valid"] + ALL_SCORE_FEATURES + ["lc_duration", "total_duration"]
    with feat_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["driver", "label", "score", "S", "L", "P"] + feat_keys)
        w.writeheader()
        score_map = {r["driver"]: r for r in sorted_results}
        for d in drivers:
            r = score_map[d]
            row = {"driver": d, "label": labels[d], "score": r["score"], "S": r["S"], "L": r["L"], "P": r["P"]}
            for k in feat_keys:
                row[k] = features[d].get(k, "")
            w.writerow(row)

    meta_fp = out_dir / "overtaking_style_feature_catalog.json"
    with meta_fp.open("w", encoding="utf-8") as f:
        json.dump({
            "axis_S_features": AXIS_S_FEATURES,
            "axis_L_features": AXIS_L_FEATURES,
            "axis_P_features": AXIS_P_FEATURES,
            "formula": "style_score = w_L * L + w_P * P - w_S * S",
            "weights": {"w_S": args.w_S, "w_L": args.w_L, "w_P": args.w_P},
            "feature_mean": mean_all,
            "feature_std": std_all,
        }, f, indent=2, ensure_ascii=False)

    if args.plot:
        _plot_heatmap(sorted_results, z_all, drivers, labels, out_dir / "overtaking_style_heatmap.png")
        _plot_scatter(sorted_results, labels, out_dir / "overtaking_style_scatter.png")

    print("[saved] {}".format(labels_fp))
    print("[saved] {}".format(proto_fp))
    print("[saved] {}".format(feat_fp))
    print("[saved] {}".format(meta_fp))
    print("[DONE]")


if __name__ == "__main__":
    main()
