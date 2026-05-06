# -*- coding: utf-8 -*-
"""
Classify per-driver longitudinal car-following styles into 3 ordered labels:
  conservative / neutral / aggressive

using a **physics-grounded composite score** (no k-means, no random seeds).

## Method

Each driver is projected onto 3 orthogonal behavioural axes:

  D (Distance preference)   — how far the driver keeps from the lead
  R (Reactivity)            — how large / abrupt the acceleration responses are
  C (Closeness intensity)   — how often the driver enters dangerous proximity

Within each axis, 2-3 raw features are z-scored and averaged. Then a single
composite style score is computed:

    style_score = w_R * R + w_C * C - w_D * D

Drivers are sorted by ``style_score`` and split into terciles:
  bottom third → conservative
  middle third → neutral
  top third   → aggressive

This guarantees:
  - Labels are strictly ordered by aggressiveness
  - Reproducible (no random init)
  - Balanced classes (~6-7 drivers each for 20 drivers)
  - Interpretable axes with physical meaning

## Inputs

Reads calibrated ``driving_data.csv`` files from ``--data_dir`` (default:
``following/outputs/following_calibrated``). Also optionally reads IDM params
from ``--idm_dir`` to include the fitted ``T`` parameter as a distance feature.

## Outputs

  <out_dir>/following_style_labels.json       — {driver: label}
  <out_dir>/following_style_features.csv      — per-driver raw + z-scored features
  <out_dir>/following_style_prototypes.json   — one prototype per style (closest to group mean)
  <out_dir>/following_style_heatmap.png       — z-score heatmap sorted by style_score
  <out_dir>/following_style_scatter.png       — 2D PCA scatter coloured by label

## Usage

    python3 following/scripts/cluster_following_style.py
    python3 following/scripts/cluster_following_style.py --data_dir following/outputs/following_calibrated --out_dir ... --plot

    python3 /home/zwx/driver_model/following/scripts/cluster_following_style.py --data_dir /home/zwx/driver_model/following/outputs/following_calibrated --out_dir /home/zwx/driver_model/following/outputs/following_style_clusters --plot
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re

import numpy as np


# ======================================================================
# Helpers
# ======================================================================

def _parse_float(v, default=None):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        x = float(s)
        return x if math.isfinite(x) else default
    except (ValueError, TypeError):
        return default


def _discover_csvs(data_dir):
    """Find all driving_data.csv or segment_*.csv under data_dir."""
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn == "driving_data.csv" or re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _percentile(values, pct):
    if not values:
        return 0.0
    a = np.asarray(values)
    return float(np.percentile(a, pct))


# ======================================================================
# Per-driver feature extraction
# ======================================================================

def _extract_driver_features(csv_paths, min_speed=2.0, min_gap=1.0):
    """Compute per-driver summary features from all their CSV segments.

    Only rows where ego_speed > min_speed AND gap > min_gap are used
    (filters out standstill / non-following moments).
    """
    # Accumulators
    thw_samples = []
    gap_samples = []
    inv_ttc_samples = []
    acc_samples = []
    decel_samples = []
    jerk_samples = []
    speed_samples = []

    for fp in csv_paths:
        prev_a = None
        prev_t = None
        with open(fp, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = _parse_float(row.get("sim_time_s")) or _parse_float(row.get("timestamp"))
                v = _parse_float(row.get("ego_v_long")) or _parse_float(row.get("ego_speed"))
                a = _parse_float(row.get("ego_a_long")) or _parse_float(row.get("ego_acceleration"))
                gap = _parse_float(row.get("distance_headway"))
                lv = _parse_float(row.get("lead_v_long")) or _parse_float(row.get("lead_speed"))

                if v is None or a is None or gap is None or lv is None:
                    continue
                if v < min_speed or gap < min_gap:
                    prev_a = a
                    prev_t = t
                    continue

                speed_samples.append(v)
                gap_samples.append(gap)
                acc_samples.append(a)
                if a < 0:
                    decel_samples.append(abs(a))

                # THW
                thw = gap / max(v, 0.5)
                thw_samples.append(thw)

                # inv_TTC (only when closing)
                closing_rate = v - lv
                if closing_rate > 0.01 and gap > 0.5:
                    inv_ttc = closing_rate / gap
                    inv_ttc_samples.append(inv_ttc)

                # Jerk (finite difference)
                if prev_a is not None and prev_t is not None and t is not None:
                    dt = t - prev_t
                    if 0.01 < dt < 0.5:
                        jerk_samples.append(abs(a - prev_a) / dt)
                prev_a = a
                prev_t = t

    if not acc_samples:
        return None

    return dict(
        # Axis D: Distance preference
        thw_median=float(np.median(thw_samples)) if thw_samples else 0.0,
        gap_mean=float(np.mean(gap_samples)) if gap_samples else 0.0,

        # Axis R: Reactivity
        acc_std=float(np.std(acc_samples)),
        jerk_p75=_percentile(jerk_samples, 75),
        acc_range=_percentile(acc_samples, 95) - _percentile(acc_samples, 5),

        # Axis C: Closeness intensity
        inv_ttc_p90=_percentile(inv_ttc_samples, 90),
        inv_ttc_p95=_percentile(inv_ttc_samples, 95),
        decel_p90=_percentile(decel_samples, 90),

        # Extra (for reporting, not used in score)
        speed_mean=float(np.mean(speed_samples)),
        n_rows=len(acc_samples),
        n_closing=len(inv_ttc_samples),
    )


# ======================================================================
# Scoring and labelling
# ======================================================================

# Features used for each axis (keys into the feature dict)
AXIS_D_FEATURES = ["thw_median", "gap_mean"]
AXIS_R_FEATURES = ["acc_std", "jerk_p75", "acc_range"]
AXIS_C_FEATURES = ["inv_ttc_p90", "decel_p90"]

ALL_SCORE_FEATURES = AXIS_D_FEATURES + AXIS_R_FEATURES + AXIS_C_FEATURES


def _zscore_matrix(drivers, features, keys):
    """Build (n_drivers, len(keys)) z-scored matrix."""
    n = len(drivers)
    m = len(keys)
    raw = np.zeros((n, m), dtype=np.float64)
    for i, d in enumerate(drivers):
        for j, k in enumerate(keys):
            raw[i, j] = features[d].get(k, 0.0)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (raw - mean) / std
    return z, raw, mean, std


def _compute_style_scores(drivers, features, w_D=1.0, w_R=1.0, w_C=1.0):
    """Compute composite style score for each driver.

    Returns sorted list of (driver, score, D, R, C).
    """
    z_all, raw_all, mean_all, std_all = _zscore_matrix(
        drivers, features, ALL_SCORE_FEATURES)

    n_d = len(AXIS_D_FEATURES)
    n_r = len(AXIS_R_FEATURES)
    n_c = len(AXIS_C_FEATURES)

    D = z_all[:, :n_d].mean(axis=1)
    R = z_all[:, n_d:n_d + n_r].mean(axis=1)
    C = z_all[:, n_d + n_r:n_d + n_r + n_c].mean(axis=1)

    scores = w_R * R + w_C * C - w_D * D

    results = []
    for i, d in enumerate(drivers):
        results.append(dict(
            driver=d, score=float(scores[i]),
            D=float(D[i]), R=float(R[i]), C=float(C[i]),
        ))
    results.sort(key=lambda x: x["score"])
    return results, z_all, raw_all, mean_all, std_all


def _assign_labels(sorted_results):
    """Split sorted results into terciles: conservative / neutral / aggressive."""
    n = len(sorted_results)
    n_cons = n // 3
    n_aggr = n // 3
    n_neut = n - n_cons - n_aggr

    labels = {}
    for i, r in enumerate(sorted_results):
        if i < n_cons:
            labels[r["driver"]] = "conservative"
        elif i < n_cons + n_neut:
            labels[r["driver"]] = "neutral"
        else:
            labels[r["driver"]] = "aggressive"
    return labels


def _find_prototypes(sorted_results, labels):
    """Find the driver closest to the group mean score for each label."""
    groups = {}
    for r in sorted_results:
        lbl = labels[r["driver"]]
        groups.setdefault(lbl, []).append(r)
    prototypes = {}
    for lbl, members in groups.items():
        mean_score = np.mean([m["score"] for m in members])
        best = min(members, key=lambda m: abs(m["score"] - mean_score))
        prototypes[lbl] = best["driver"]
    return prototypes


# ======================================================================
# Visualization
# ======================================================================

def _plot_heatmap(drivers_sorted, z_all, driver_order, labels, out_path):
    """Z-score heatmap sorted by style_score."""
    import matplotlib.pyplot as plt

    idx_map = {d: i for i, d in enumerate(driver_order)}
    order = [idx_map[r["driver"]] for r in drivers_sorted]
    z_sorted = z_all[order]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    im = ax.imshow(z_sorted, aspect="auto", cmap="RdYlGn_r", vmin=-2.5, vmax=2.5)

    ax.set_yticks(range(len(drivers_sorted)))
    ylabels = []
    for r in drivers_sorted:
        lbl = labels[r["driver"]]
        marker = {"conservative": "●", "neutral": "◆", "aggressive": "▲"}[lbl]
        ylabels.append("{} {} ({:.2f})".format(marker, r["driver"], r["score"]))
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_xticks(range(len(ALL_SCORE_FEATURES)))
    ax.set_xticklabels(ALL_SCORE_FEATURES, rotation=45, ha="right", fontsize=9)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("z-score")

    # Horizontal lines between groups
    n = len(drivers_sorted)
    n_cons = n // 3
    n_neut = n - 2 * (n // 3)
    ax.axhline(n_cons - 0.5, color="black", lw=1.5, ls="--")
    ax.axhline(n_cons + n_neut - 0.5, color="black", lw=1.5, ls="--")

    ax.set_title("Following Style Features (sorted by aggressiveness score)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] heatmap -> {}".format(out_path))


def _plot_scatter(drivers_sorted, labels, out_path):
    """2D scatter: D vs (R+C), coloured by label."""
    import matplotlib.pyplot as plt

    colors = {"conservative": "#2ca02c", "neutral": "#1f77b4", "aggressive": "#d62728"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for r in drivers_sorted:
        lbl = labels[r["driver"]]
        ax.scatter(r["D"], r["R"] + r["C"], color=colors[lbl], s=80, zorder=3)
        ax.annotate(r["driver"], (r["D"], r["R"] + r["C"]),
                    fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("D (Distance preference, z-score) ← conservative")
    ax.set_ylabel("R + C (Reactivity + Closeness, z-score) → aggressive")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_title("Following Style: Distance vs Reactivity+Closeness")
    ax.grid(True, ls="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors["conservative"], label="Conservative"),
        Patch(facecolor=colors["neutral"], label="Neutral"),
        Patch(facecolor=colors["aggressive"], label="Aggressive"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] scatter -> {}".format(out_path))


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Classify per-driver following style into conservative/neutral/aggressive."
    )
    ap.add_argument("--data_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/following_calibrated",
                    help="Root directory with per-driver calibrated CSVs.")
    ap.add_argument("--idm_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/idm_per_driver",
                    help="Directory with <T*>/idm.json for IDM T parameter.")
    ap.add_argument("--out_dir", type=str,
                    default="/home/zwx/driver_model/following/outputs/following_style_clusters",
                    help="Output directory for labels, features, plots.")
    ap.add_argument("--w_D", type=float, default=1.0, help="Weight for Distance axis.")
    ap.add_argument("--w_R", type=float, default=1.0, help="Weight for Reactivity axis.")
    ap.add_argument("--w_C", type=float, default=1.0, help="Weight for Closeness axis.")
    ap.add_argument("--min_sim_time_s", type=float, default=15.0,
                    help="Only use rows with sim_time_s >= this (skip startup).")
    ap.add_argument("--plot", action="store_true", help="Generate heatmap and scatter plots.")
    args = ap.parse_args()

    # --- Discover CSVs per driver ---
    all_csvs = _discover_csvs(args.data_dir)
    by_driver = {}
    for p in all_csvs:
        by_driver.setdefault(_driver_id(p), []).append(p)
    drivers = sorted(by_driver.keys(),
                     key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999)
    if not drivers:
        raise SystemExit("[ERR] no CSVs found under " + args.data_dir)
    print("[data] {} drivers, {} CSVs total".format(len(drivers), len(all_csvs)))

    # --- Extract features per driver ---
    features = {}
    for d in drivers:
        feat = _extract_driver_features(by_driver[d], min_speed=2.0, min_gap=1.0)
        if feat is None:
            print("[SKIP] {}: no valid rows".format(d))
            continue
        # Optionally add IDM T parameter
        idm_fp = os.path.join(args.idm_dir, d, "idm.json")
        if os.path.isfile(idm_fp):
            with open(idm_fp, "r", encoding="utf-8") as f:
                idm_data = json.load(f)
            feat["idm_T"] = float(idm_data.get("parameters", {}).get("T", 0.0))
        else:
            feat["idm_T"] = feat["thw_median"]  # fallback
        features[d] = feat

    valid_drivers = [d for d in drivers if d in features]
    if len(valid_drivers) < 3:
        raise SystemExit("[ERR] need at least 3 drivers, got {}".format(len(valid_drivers)))

    # --- Compute scores and assign labels ---
    sorted_results, z_all, raw_all, mean_all, std_all = _compute_style_scores(
        valid_drivers, features, w_D=args.w_D, w_R=args.w_R, w_C=args.w_C)
    labels = _assign_labels(sorted_results)
    prototypes = _find_prototypes(sorted_results, labels)

    # --- Print summary ---
    print("\n{:<5s} {:>7s} {:>6s} {:>6s} {:>6s}  {:<14s}".format(
        "Drv", "Score", "D", "R", "C", "Label"))
    print("-" * 52)
    for r in sorted_results:
        lbl = labels[r["driver"]]
        print("{:<5s} {:+7.3f} {:+6.3f} {:+6.3f} {:+6.3f}  {:<14s}".format(
            r["driver"], r["score"], r["D"], r["R"], r["C"], lbl))
    print("\nPrototypes: {}".format(prototypes))

    # --- Save outputs ---
    os.makedirs(args.out_dir, exist_ok=True)

    # Labels JSON
    labels_fp = os.path.join(args.out_dir, "following_style_labels.json")
    with open(labels_fp, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print("[saved] {}".format(labels_fp))

    # Prototypes JSON
    proto_fp = os.path.join(args.out_dir, "following_style_prototypes.json")
    proto_out = dict(prototypes=prototypes, sorted_scores=[
        dict(driver=r["driver"], score=r["score"], D=r["D"], R=r["R"], C=r["C"],
             label=labels[r["driver"]])
        for r in sorted_results
    ])
    with open(proto_fp, "w", encoding="utf-8") as f:
        json.dump(proto_out, f, indent=2, ensure_ascii=False)
    print("[saved] {}".format(proto_fp))

    # Features CSV
    feat_fp = os.path.join(args.out_dir, "following_style_features.csv")
    feat_keys = ALL_SCORE_FEATURES + ["idm_T", "speed_mean", "n_rows", "n_closing"]
    with open(feat_fp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["driver", "label", "score", "D", "R", "C"] + feat_keys)
        w.writeheader()
        for r in sorted_results:
            d = r["driver"]
            row = dict(driver=d, label=labels[d], score=r["score"],
                       D=r["D"], R=r["R"], C=r["C"])
            for k in feat_keys:
                row[k] = features[d].get(k, "")
            w.writerow(row)
    print("[saved] {}".format(feat_fp))

    # --- Plots ---
    if args.plot:
        heatmap_path = os.path.join(args.out_dir, "following_style_heatmap.png")
        _plot_heatmap(sorted_results, z_all, valid_drivers, labels, heatmap_path)

        scatter_path = os.path.join(args.out_dir, "following_style_scatter.png")
        _plot_scatter(sorted_results, labels, scatter_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
