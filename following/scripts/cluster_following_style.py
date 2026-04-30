# -*- coding: utf-8 -*-
"""
Cluster per-driver longitudinal car-following styles into:
  conservative / neutral / aggressive.

Each driver is summarized with **gap / speed-difference / acceleration-difference**
statistics (longitudinal interaction with the lead), then k-means (k=3) on z-scored
features. Pedal-only signals are not used for clustering.

Preferred input:
  outputs/following_il_clean_gap04 produced from the latest calibrated data.

Row-level fields used (with fallbacks for older CSVs):
  distance_headway, time_headway
  relative_v_long  (else lead_speed - ego_speed; uses ego_v_long / lead_v_long when present)
  ego_a_long       (else ego_acceleration)
  lead_a_long      (else lead_acceleration)
  acc_diff := lead_long - ego_long   (summarized per driver)

Also writes following_style_prototypes.json: one prototype driver per style (closest to
centroid in z-scored cluster feature space) for typical lateral residual replay.

Visualization (optional):
  python3 cluster_following_style.py --plot
  -> saves PCA 2D scatter to --out_dir/following_style_clusters_pca.png
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import re

import numpy as np


def _parse_float(v, default=None):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        x = float(s)
        if math.isfinite(x):
            return x
        return default
    except ValueError:
        return default


def _discover_segments(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _percentile(values, pct):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((pct / 100.0) * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def _mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def _std(values):
    if not values:
        return 0.0
    m = _mean(values)
    return (sum((x - m) ** 2 for x in values) / float(len(values))) ** 0.5


def _value(row, primary, fallback=None):
    v = _parse_float(row.get(primary))
    if v is not None:
        return v
    if fallback is not None:
        return _parse_float(row.get(fallback), 0.0)
    return 0.0


def _row_features(row):
    ego_v = _value(row, "ego_v_long", "ego_speed")
    lead_v = _value(row, "lead_v_long", "lead_speed")
    rel_v = _parse_float(row.get("relative_v_long"))
    if rel_v is None:
        rel_v = lead_v - ego_v
    ego_a = _value(row, "ego_a_long", "ego_acceleration")
    lead_a = _value(row, "lead_a_long", "lead_acceleration")
    acc_diff = lead_a - ego_a
    headway = _parse_float(row.get("distance_headway"))
    time_headway = _parse_float(row.get("time_headway"))
    throttle = _parse_float(row.get("throttle"), 0.0)
    brake = _parse_float(row.get("brake"), 0.0)
    return {
        "ego_v": ego_v,
        "lead_v": lead_v,
        "rel_v": rel_v,
        "ego_a": ego_a,
        "lead_a": lead_a,
        "acc_diff": acc_diff,
        "headway": headway,
        "time_headway": time_headway,
        "throttle": throttle,
        "brake": brake,
    }


def _summarize_driver(paths):
    ego_v = []
    rel_v_signed = []
    rel_v_abs = []
    acc = []
    acc_abs = []
    acc_diff = []
    acc_diff_abs = []
    headway = []
    time_headway = []
    throttle = []
    brake = []

    rows = 0
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = _row_features(row)
                rows += 1
                ego_v.append(r["ego_v"])
                rel_v_signed.append(r["rel_v"])
                rel_v_abs.append(abs(r["rel_v"]))
                acc.append(r["ego_a"])
                acc_abs.append(abs(r["ego_a"]))
                acc_diff.append(r["acc_diff"])
                acc_diff_abs.append(abs(r["acc_diff"]))
                if r["headway"] is not None and r["headway"] < 300:
                    headway.append(r["headway"])
                if (
                    r["time_headway"] is not None
                    and r["time_headway"] < 50
                    and abs(r["time_headway"] - 999.0) > 1e-6
                ):
                    time_headway.append(r["time_headway"])
                throttle.append(r["throttle"])
                brake.append(r["brake"])

    positive_acc = [x for x in acc if x > 0.2]
    negative_acc_abs = [abs(x) for x in acc if x < -0.2]
    brake_active = [1.0 if x > 0.02 else 0.0 for x in brake]
    throttle_active = [1.0 if x > 0.05 else 0.0 for x in throttle]

    return {
        "n_rows": rows,
        "n_segments": len(paths),
        "ego_v_mean": _mean(ego_v),
        "ego_v_p85": _percentile(ego_v, 85),
        "headway_median": _percentile(headway, 50),
        "headway_p25": _percentile(headway, 25),
        "time_headway_median": _percentile(time_headway, 50),
        "time_headway_p25": _percentile(time_headway, 25),
        "relative_v_mean": _mean(rel_v_signed),
        "relative_v_std": _std(rel_v_signed),
        "relative_v_abs_mean": _mean(rel_v_abs),
        "relative_v_abs_p95": _percentile(rel_v_abs, 95),
        "acc_diff_mean": _mean(acc_diff),
        "acc_diff_abs_mean": _mean(acc_diff_abs),
        "acc_diff_abs_p95": _percentile(acc_diff_abs, 95),
        "acc_mean": _mean(acc),
        "acc_std": _std(acc),
        "abs_acc_p95": _percentile(acc_abs, 95),
        "positive_acc_mean": _mean(positive_acc),
        "decel_abs_mean": _mean(negative_acc_abs),
        "throttle_mean": _mean(throttle),
        "throttle_active_ratio": _mean(throttle_active),
        "brake_mean": _mean(brake),
        "brake_active_ratio": _mean(brake_active),
    }


def _zscore_matrix(rows, feature_names):
    cols = []
    for name in feature_names:
        col = [r[name] for r in rows]
        m = _mean(col)
        s = _std(col)
        if s < 1e-9:
            s = 1.0
        cols.append((m, s))
    mat = []
    for r in rows:
        mat.append([(r[name] - cols[i][0]) / cols[i][1] for i, name in enumerate(feature_names)])
    return mat


def _kmeans(points, k, seed, max_iter=100):
    rng = random.Random(seed)
    if len(points) < k:
        raise RuntimeError("Need at least {} drivers for clustering.".format(k))
    centers = [list(p) for p in rng.sample(points, k)]
    labels = [0] * len(points)
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            dists = [
                sum((p[j] - c[j]) ** 2 for j in range(len(p)))
                for c in centers
            ]
            lab = min(range(k), key=lambda x: dists[x])
            if labels[i] != lab:
                labels[i] = lab
                changed = True
        new_centers = []
        for lab in range(k):
            members = [points[i] for i in range(len(points)) if labels[i] == lab]
            if not members:
                new_centers.append(list(rng.choice(points)))
            else:
                new_centers.append([
                    sum(p[j] for p in members) / float(len(members))
                    for j in range(len(points[0]))
                ])
        centers = new_centers
        if not changed:
            break
    return labels, centers


def _pca2(zpoints):
    """PCA to 2D on row-wise centered z-scored matrix. Returns (n,2) coords and variance fractions."""
    X = np.asarray(zpoints, dtype=np.float64)
    if X.shape[0] < 2:
        return None, None
    X = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:2].T
    proj = X @ W
    tot = float(np.sum(s ** 2)) + 1e-12
    var_frac = [(s[i] ** 2 / tot) for i in range(min(2, len(s)))]
    return proj, var_frac


def _style_prototype_summary(rows, points, cluster_features):
    """One prototype driver per style = closest to cluster centroid in z-scored feature space."""
    by_style = {}
    for i, r in enumerate(rows):
        sty = r["style_label"]
        by_style.setdefault(sty, []).append(i)
    summary = {}
    for sty in sorted(by_style.keys()):
        idxs = by_style[sty]
        pts = np.asarray([points[i] for i in idxs], dtype=np.float64)
        c = pts.mean(axis=0)
        d2 = np.sum((pts - c) ** 2, axis=1)
        j = int(np.argmin(d2))
        proto_i = idxs[j]
        summary[sty] = {
            "prototype_driver": rows[proto_i]["driver_id"],
            "drivers_in_style": [rows[i]["driver_id"] for i in idxs],
            "n_drivers": len(idxs),
            "cluster_id_numeric": int(rows[proto_i]["cluster_id"]),
            "prototype_features": {k: rows[proto_i].get(k) for k in cluster_features},
        }
    return summary


def _save_cluster_plot(rows, zpoints, out_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skip figure. Try: pip install matplotlib")
        return

    xy, var_frac = _pca2(zpoints)
    if xy is None:
        print("[WARN] Need at least 2 drivers for PCA plot.")
        return

    colors = {
        "conservative": "#27ae60",
        "neutral": "#3498db",
        "aggressive": "#e74c3c",
    }
    fig, ax = plt.subplots(figsize=(9, 7))
    for style in ("conservative", "neutral", "aggressive"):
        idx = [i for i, r in enumerate(rows) if r.get("style_label") == style]
        if not idx:
            continue
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=120,
            c=colors[style],
            label=style,
            edgecolors="0.3",
            linewidths=0.6,
            zorder=2,
        )
    for i, r in enumerate(rows):
        ax.annotate(
            r.get("driver_id", str(i)),
            (xy[i, 0], xy[i, 1]),
            fontsize=9,
            xytext=(4, 3),
            textcoords="offset points",
            color="0.15",
        )
    pct0 = 100.0 * var_frac[0] if var_frac else 0.0
    pct1 = 100.0 * var_frac[1] if len(var_frac) > 1 else 0.0
    ax.set_xlabel("PC1 ({:.0f}% variance)".format(pct0))
    ax.set_ylabel("PC2 ({:.0f}% variance)".format(pct1))
    ax.set_title("Car-following style clusters (PCA on z-scored gap / Δv / Δa features)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("[OK] figure:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/following_il_clean_gap04",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/following_style_clusters",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Save PCA 2D scatter (matplotlib) to --plot_path",
    )
    ap.add_argument(
        "--plot_path",
        type=str,
        default="",
        help="PNG path (default: <out_dir>/following_style_clusters_pca.png)",
    )
    args = ap.parse_args()

    paths = _discover_segments(args.data_dir)
    by_driver = {}
    for p in paths:
        by_driver.setdefault(_driver_id(p), []).append(p)

    rows = []
    for driver in sorted(by_driver.keys(), key=lambda x: int(x[1:]) if x.startswith("T") and x[1:].isdigit() else 999):
        metrics = _summarize_driver(by_driver[driver])
        metrics["driver_id"] = driver
        rows.append(metrics)

    # k-means on **interaction / difference** summaries only (not absolute ego speed or pedals).
    cluster_features = [
        "headway_median",
        "headway_p25",
        "time_headway_median",
        "time_headway_p25",
        "relative_v_abs_mean",
        "relative_v_std",
        "acc_diff_abs_mean",
        "acc_diff_abs_p95",
    ]
    points = _zscore_matrix(rows, cluster_features)
    labels, _ = _kmeans(points, 3, args.seed)

    # Rank clusters: conservative = larger gaps/time gaps, smaller |Δv| and |Δa| activity.
    cluster_scores = {}
    for lab in range(3):
        idxs = [i for i, x in enumerate(labels) if x == lab]
        if not idxs:
            cluster_scores[lab] = 0.0
            continue
        score = 0.0
        for i in idxs:
            p = points[i]
            f = dict(zip(cluster_features, p))
            score += (
                -1.0 * f["time_headway_median"]
                -1.0 * f["time_headway_p25"]
                -1.0 * f["headway_median"]
                -0.9 * f["headway_p25"]
                +1.0 * f["relative_v_abs_mean"]
                +0.8 * f["relative_v_std"]
                +1.0 * f["acc_diff_abs_mean"]
                +1.0 * f["acc_diff_abs_p95"]
            )
        cluster_scores[lab] = score / float(len(idxs))
    ordered = sorted(cluster_scores.keys(), key=lambda x: cluster_scores[x])
    label_name = {
        ordered[0]: "conservative",
        ordered[1]: "neutral",
        ordered[2]: "aggressive",
    }

    for i, r in enumerate(rows):
        r["cluster_id"] = labels[i]
        r["style_label"] = label_name[labels[i]]

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp = os.path.join(args.out_dir, "driver_following_style_clusters.csv")
    fieldnames = ["driver_id", "style_label", "cluster_id"] + [
        "n_segments",
        "n_rows",
    ] + cluster_features + [
        "relative_v_mean",
        "relative_v_abs_p95",
        "acc_diff_mean",
        "ego_v_mean",
        "acc_mean",
        "abs_acc_p95",
        "acc_std",
        "throttle_active_ratio",
        "brake_mean",
    ]
    with open(out_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    proto_path = os.path.join(args.out_dir, "following_style_prototypes.json")
    proto_summary = _style_prototype_summary(rows, points, cluster_features)
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cluster_features_zscored_pca_order": cluster_features,
                "styles": proto_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[OK] prototypes:", proto_path)

    print("[OK] drivers:", len(rows))
    print("[OK] output:", out_fp)
    for r in rows:
        print("{}: {}".format(r["driver_id"], r["style_label"]))

    if args.plot:
        plot_fp = args.plot_path.strip()
        if not plot_fp:
            plot_fp = os.path.join(args.out_dir, "following_style_clusters_pca.png")
        _save_cluster_plot(rows, points, plot_fp)


if __name__ == "__main__":
    main()
