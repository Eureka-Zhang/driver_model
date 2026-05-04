# -*- coding: utf-8 -*-
"""
Cluster per-driver longitudinal car-following styles into:
  conservative / neutral / aggressive.

Each driver is summarized with **gap / speed-difference / acceleration-difference**
statistics (longitudinal interaction with the lead), then k-means (k=3) on z-scored
features. Pedal-only signals are not used for clustering.

Preferred input:
  outputs/following_il_clean_gap04 produced from the latest calibrated data.

Row-level longitudinal signals:
  ego / lead speeds (ego_v_long or ego_speed, etc.)
  distance_headway, time_headway (time_headway may be missing → computed from headway/max(ego_v, eps))
  TTC **not** taken from CSV: ``headway/(ego_v − lead_v)`` when ego is catching up (>1e−2 m/s).
  Clustering uses **percentiles of** ``1/time_headway`` and ``1/TTC`` (1/s-style tightness / closing intensity),
  not raw THW/TTC durations.
  jerk (longitudinal ``ego_a_long`` / ``ego_acceleration`` differentiated w.r.t. ``timestamp``, |j| summarized).

K-means uses **weighted** squared distance on z-scored features:
  sum_j w_j (z_ij - c_kj)^2
See ``--cluster_dim_weights`` (default aligns with ``CLUSTER_FEATURES_ENHANCED`` / ``CLUSTER_FEATURES``).
Style naming (conservative / neutral / aggressive) uses a separate z-score heuristic with
**positive-only** coefficients (conservative cues enter as ``-z``).

Also writes following_style_prototypes.json: one prototype driver per style (closest to
centroid in the same weighted metric) for typical lateral residual replay.

Visualization (optional):
  python3 cluster_following_style.py --plot
  -> saves PCA 2D scatter to --out_dir/following_style_clusters_pca.png
  
  python3 following/scripts/cluster_following_style.py \
    --data_dir following/outputs/following_il_clean_gap04 \
    --out_dir following/outputs/following_style_clusters \
    --plot \
    --cluster_dim_weights 1,1,1,1,1,1,1,1,1,1,1,1 \
    --plot_path following/outputs/following_style_clusters/my_pca.png
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


def _var(values):
    s = _std(values)
    return s * s


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


def _thw_csv_usable(th_csv):
    if th_csv is None:
        return False
    if not math.isfinite(float(th_csv)):
        return False
    if abs(float(th_csv) - 999.0) <= 1e-3:
        return False
    return float(th_csv) > 1e-4 and float(th_csv) < 500.0


def _effective_time_headway_csv_or_compute(row, headway_m, ego_v):
    """Use CSV ``time_headway`` when usable; otherwise ``distance_headway / ego_v`` (mirror collector)."""
    th_raw = _parse_float(row.get("time_headway"))
    if _thw_csv_usable(th_raw):
        return float(th_raw)
    if headway_m is None or ego_v < 0.5:
        return None
    if headway_m < 300.0:
        return float(headway_m) / float(ego_v)
    return None


def _closing_ttc_from_longitudinal(distance_m, ego_v_long, lead_v_long):
    """
    Longitudinal TTC from kinematics (**not CSV ``ttc``**):

    ``distance_headway / (ego_v − lead_v)`` when ego is approaching the lead (>1e−2 m/s).
    Mirrors ``replay/experiment.DataCollector`` logic.
    """
    if distance_m is None:
        return None
    dh = float(distance_m)
    if not math.isfinite(dh) or dh <= 1e-3 or dh >= 300.0:
        return None
    dv = float(ego_v_long) - float(lead_v_long)
    if not math.isfinite(dv) or dv <= 1e-2:
        return None
    t = dh / dv
    if not math.isfinite(t) or t <= 0.0:
        return None
    return min(t, 999.0)


def _central_derivative(times, vals):
    n = len(times)
    if n == 0:
        return []
    out = [0.0] * n
    if n >= 3:
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 1e-12:
                out[i] = (vals[i + 1] - vals[i - 1]) / dt
    if n >= 2:
        dt = times[1] - times[0]
        if dt > 1e-12:
            out[0] = (vals[1] - vals[0]) / dt
        dt = times[-1] - times[-2]
        if dt > 1e-12:
            out[-1] = (vals[-1] - vals[-2]) / dt
    return out


def _summarize_driver(paths):
    ego_v = []
    rel_v_signed = []
    rel_v_abs = []
    rel_v_gap_ego_minus_lead = []
    acc = []
    acc_abs = []
    acc_diff = []
    acc_diff_abs = []
    headway = []
    time_headway_effective = []
    throttle = []
    brake = []
    jerk_abs_samples = []

    ttc_positive_closing_samples = []

    rows = 0
    thw_known_count = 0
    thw_lt_1s_count = 0

    for fp in paths:
        seg_times = []
        seg_acc = []

        with open(fp, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ts = _parse_float(row.get("timestamp"))
                r = _row_features(row)

                ego_v.append(r["ego_v"])
                rel_v_signed.append(r["rel_v"])
                rel_v_abs.append(abs(r["rel_v"]))
                rel_v_gap_ego_minus_lead.append(float(r["ego_v"]) - float(r["lead_v"]))

                acc.append(r["ego_a"])
                acc_abs.append(abs(r["ego_a"]))
                acc_diff.append(r["acc_diff"])
                acc_diff_abs.append(abs(r["acc_diff"]))

                hw = r["headway"]

                thaw = None
                if hw is not None and hw < 300.0:
                    headway.append(hw)
                    thaw = _effective_time_headway_csv_or_compute(row, hw, float(r["ego_v"]))

                if thaw is not None and math.isfinite(thaw):
                    time_headway_effective.append(thaw)
                    thw_known_count += 1
                    if thaw < 1.0:
                        thw_lt_1s_count += 1

                tty = None
                if hw is not None:
                    tty = _closing_ttc_from_longitudinal(hw, float(r["ego_v"]), float(r["lead_v"]))
                if tty is not None:
                    ttc_positive_closing_samples.append(tty)

                throttle.append(r["throttle"])
                brake.append(r["brake"])
                rows += 1

                if ts is not None:
                    seg_times.append(ts)
                    seg_acc.append(float(r["ego_a"]))

        if len(seg_times) >= 2:
            jerk_vec = _central_derivative(seg_times, seg_acc)
            for ji in jerk_vec:
                if math.isfinite(float(ji)):
                    jerk_abs_samples.append(abs(float(ji)))

    positive_acc = [x for x in acc if x > 0.2]
    negative_acc_abs = [abs(x) for x in acc if x < -0.2]
    brake_active = [1.0 if x > 0.02 else 0.0 for x in brake]
    throttle_active = [1.0 if x > 0.05 else 0.0 for x in throttle]

    thw_less_1s_ratio = float(thw_lt_1s_count) / float(thw_known_count) if thw_known_count > 0 else 0.0

    inv_time_headway_samples = []
    for t in time_headway_effective:
        if t is not None and float(t) > 1e-9:
            inv_time_headway_samples.append(1.0 / float(t))

    inv_ttc_samples = []
    for tty in ttc_positive_closing_samples:
        if tty is not None and float(tty) > 1e-9:
            inv_ttc_samples.append(1.0 / float(tty))

    # Legacy single-stat export (seconds); clustering uses inverse percentiles below.
    if ttc_positive_closing_samples:
        ttc_p05 = _percentile(ttc_positive_closing_samples, 5)
    else:
        ttc_p05 = 999.0

    jerk_abs_p75 = _percentile(jerk_abs_samples, 75) if jerk_abs_samples else 0.0
    rel_v_std = _std(rel_v_gap_ego_minus_lead)

    return {
        "n_rows": rows,
        "n_segments": len(paths),
        "ego_v_mean": _mean(ego_v),
        "ego_v_var": _var(ego_v),
        "ego_v_p85": _percentile(ego_v, 85),
        "headway_mean": _mean(headway),
        "headway_median": _percentile(headway, 50),
        "headway_p25": _percentile(headway, 25),
        "time_headway_median": _percentile(time_headway_effective, 50),
        "time_headway_p25": _percentile(time_headway_effective, 25),
        "inv_time_headway_p25": _percentile(inv_time_headway_samples, 25),
        "inv_time_headway_median": _percentile(inv_time_headway_samples, 50),
        "thw_less_1s_ratio": float(thw_less_1s_ratio),
        "ttc_p05": float(ttc_p05),
        "inv_ttc_p50": _percentile(inv_ttc_samples, 50),
        "inv_ttc_p95": _percentile(inv_ttc_samples, 95),
        "relative_v_mean": _mean(rel_v_signed),
        "relative_v_std": _std(rel_v_signed),
        "relative_v_abs_mean": _mean(rel_v_abs),
        "relative_v_abs_p95": _percentile(rel_v_abs, 95),
        "acc_diff_mean": _mean(acc_diff),
        "acc_diff_abs_mean": _mean(acc_diff_abs),
        "acc_diff_abs_p95": _percentile(acc_diff_abs, 95),
        "acc_mean": _mean(acc),
        "acc_std": _std(acc),
        "acc_var": _var(acc),
        "abs_acc_p95": _percentile(acc_abs, 95),
        "accel_abs_median": _percentile(positive_acc, 50),
        "accel_abs_p75": _percentile(positive_acc, 75),
        "decel_abs_median": _percentile(negative_acc_abs, 50),
        "decel_abs_p75": _percentile(negative_acc_abs, 75),
        "jerk_abs_p75": float(jerk_abs_p75),
        "rel_v_std": float(rel_v_std),
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


def _kmeans(points, k, seed, max_iter=100, dim_weights=None):
    """Lloyd on weighted SSE: sum_j w_j (p_j - c_j)^2. Centroids remain coordinate-wise means."""
    rng = random.Random(seed)
    if len(points) < k:
        raise RuntimeError("Need at least {} drivers for clustering.".format(k))
    d = len(points[0])
    if dim_weights is None:
        w = [1.0] * d
    else:
        w = list(dim_weights)
        if len(w) != d:
            raise RuntimeError(
                "dim_weights length {} != feature dim {}".format(len(w), d)
            )
        if any(x <= 0.0 for x in w):
            raise RuntimeError("dim_weights must be positive")
    centers = [list(p) for p in rng.sample(points, k)]
    labels = [0] * len(points)
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            dists = [
                sum(w[j] * (p[j] - c[j]) ** 2 for j in range(d))
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
                    for j in range(d)
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


CLUSTER_FEATURES_ENHANCED = [
    "headway_median",
    "headway_p25",
    "inv_time_headway_p25",
    "inv_time_headway_median",
    "inv_ttc_p50",
    "inv_ttc_p95",
    "ego_v_var",
    "accel_abs_median",
    "accel_abs_p75",
    "decel_abs_median",
    "decel_abs_p75",
    "jerk_abs_p75",
]


CLUSTER_FEATURES = CLUSTER_FEATURES_ENHANCED


def parse_cluster_dim_weights(s):
    """Parse comma-separated positive weights; length must match ``CLUSTER_FEATURES``."""
    dim_weights = [float(x.strip()) for x in s.split(",") if str(x).strip()]
    if len(dim_weights) != len(CLUSTER_FEATURES):
        raise RuntimeError(
            "cluster_dim_weights: expected {} values ({}), got {}".format(
                len(CLUSTER_FEATURES),
                ",".join(CLUSTER_FEATURES),
                len(dim_weights),
            )
        )
    if any(w <= 0.0 for w in dim_weights):
        raise RuntimeError("cluster_dim_weights must all be positive")
    return dim_weights


def assign_kmeans_styles(rows, dim_weights, seed):
    """
    Run z-score + weighted k-means (k=3) and assign conservative/neutral/aggressive.

    Mutates each row in ``rows`` with keys cluster_id, style_label.
    Each row must include driver_id and all CLUSTER_FEATURES fields (raw metrics, not z).

    Returns:
        dict with keys: points (z-scored rows), cluster_features, numeric_cluster_to_style
    """
    cluster_features = list(CLUSTER_FEATURES)
    if len(dim_weights) != len(cluster_features):
        raise RuntimeError(
            "dim_weights length {} != {}".format(len(dim_weights), len(cluster_features))
        )
    if any(w <= 0.0 for w in dim_weights):
        raise RuntimeError("dim_weights must be positive")
    if len(rows) < 3:
        raise RuntimeError("Need at least 3 drivers for k=3 clustering.")

    points = _zscore_matrix(rows, cluster_features)
    labels, _ = _kmeans(points, 3, seed, dim_weights=dim_weights)

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
            # Aggressiveness ↑: all coefficients positive (conservative-oriented z features enter as -z).
            score += (
                1.0 * (-f["headway_median"])
                + 0.95 * (-f["headway_p25"])
                + 0.9 * f["inv_time_headway_p25"]
                + 1.0 * f["inv_time_headway_median"]
                + 1.0 * f["inv_ttc_p50"]
                + 0.95 * f["inv_ttc_p95"]
                + 1.0 * f["ego_v_var"]
                + 0.5 * f["accel_abs_median"]
                + 0.5 * f["accel_abs_p75"]
                + 0.5 * f["decel_abs_median"]
                + 0.5 * f["decel_abs_p75"]
                + 0.5 * f["jerk_abs_p75"]
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

    return {
        "points": points,
        "cluster_features": cluster_features,
        "numeric_cluster_to_style": label_name,
    }


def _style_prototype_summary(rows, points, cluster_features, dim_weights=None):
    """One prototype driver per style = closest to cluster centroid (same metric as k-means)."""
    w = np.ones(len(cluster_features), dtype=np.float64)
    if dim_weights is not None:
        w = np.asarray(dim_weights, dtype=np.float64)
        if w.shape[0] != len(cluster_features):
            raise RuntimeError("dim_weights length mismatch in prototype summary")
    by_style = {}
    for i, r in enumerate(rows):
        sty = r["style_label"]
        by_style.setdefault(sty, []).append(i)
    summary = {}
    for sty in sorted(by_style.keys()):
        idxs = by_style[sty]
        pts = np.asarray([points[i] for i in idxs], dtype=np.float64)
        c = pts.mean(axis=0)
        d2 = np.sum(w * (pts - c) ** 2, axis=1)
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
    ax.set_title("Car-following style clusters (PCA on z-scored CLUSTER_FEATURES_ENHANCED)")
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
    ap.add_argument(
        "--cluster_dim_weights",
        type=str,
        default="1,1,1,1,1,1,1,1,1,1,1,1",
        help=(
            "Comma-separated positive weights for k-means / prototype distance on z-scored "
            "features, order: "
            + ", ".join(CLUSTER_FEATURES)
        ),
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

    cluster_features = list(CLUSTER_FEATURES)
    dim_weights = parse_cluster_dim_weights(args.cluster_dim_weights)
    print(
        "[INFO] k-means dim weights:",
        ", ".join("{}={}".format(n, w) for n, w in zip(cluster_features, dim_weights)),
    )

    result = assign_kmeans_styles(rows, dim_weights, args.seed)
    points = result["points"]

    os.makedirs(args.out_dir, exist_ok=True)
    out_fp = os.path.join(args.out_dir, "driver_following_style_clusters.csv")
    fieldnames = ["driver_id", "style_label", "cluster_id"] + [
        "n_segments",
        "n_rows",
    ] + cluster_features + [
        "time_headway_median",
        "time_headway_p25",
        "ttc_p05",
        "thw_less_1s_ratio",
        "rel_v_std",
        "headway_mean",
        "acc_var",
        "relative_v_std",
        "positive_acc_mean",
        "decel_abs_mean",
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
    proto_summary = _style_prototype_summary(
        rows, points, cluster_features, dim_weights=dim_weights
    )
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cluster_features_zscored_pca_order": cluster_features,
                "cluster_dim_weights": dict(zip(cluster_features, dim_weights)),
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
