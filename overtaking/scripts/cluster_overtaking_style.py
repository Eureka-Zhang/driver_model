# -*- coding: utf-8 -*-
"""
Cluster per-driver overtaking styles (conservative / neutral / aggressive).

Features summarize lane usage (right->left->right), lateral motion, and gap-related
extremes during overtaking segments (``segment_*.csv`` under --data_dir).

Lane bands default to the same as ``extract_overtaking_phases.classify_lane``.

Example::

  python3 overtaking/scripts/cluster_overtaking_style.py \\
    --data_dir overtaking/outputs/overtaking_il_clean_gap04 \\
    --out_dir overtaking/outputs/overtaking_style_clusters \\
    --plot --seed 42
"""
from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CFS_PATH = os.path.join(_REPO_ROOT, "following", "scripts", "cluster_following_style.py")
_EOP_PATH = os.path.join(_REPO_ROOT, "following", "scripts", "extract_overtaking_phases.py")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfs = _load_module("cluster_following_style", _CFS_PATH)
_eop = _load_module("extract_overtaking_phases", _EOP_PATH)

_parse_float = _cfs._parse_float
_mean = _cfs._mean
_std = _cfs._std
_var = _cfs._var
_percentile = _cfs._percentile
_zscore_matrix = _cfs._zscore_matrix
_kmeans = _cfs._kmeans
_pca2 = _cfs._pca2
classify_lane = _eop.classify_lane


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


def _summarize_driver_overtaking(paths, left_y_min, left_y_max, right_y_min, right_y_max):
    left_n = 0
    total = 0
    ys = []
    v_lats = []
    y_rates = []
    thws = []
    ttcs = []
    acc_abs = []
    speeds = []

    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        prev_t = None
        prev_y = None
        for row in rows:
            y = _parse_float(row.get("ego_pos_y"))
            if y is None:
                continue
            lane = classify_lane(y, left_y_min, left_y_max, right_y_min, right_y_max)
            if lane == "L":
                left_n += 1
            total += 1
            ys.append(y)

            ts = _parse_float(row.get("timestamp"))
            vl = _parse_float(row.get("ego_v_lat"))
            if vl is not None:
                v_lats.append(abs(vl))
            if ts is not None and prev_t is not None and prev_y is not None and y is not None:
                dt = ts - prev_t
                if dt > 1e-6:
                    y_rates.append(abs((y - prev_y) / dt))
            prev_t, prev_y = ts, y

            th = _parse_float(row.get("time_headway"))
            if th is not None and abs(th - 999.0) > 1e-6 and th < 80:
                thws.append(th)
            tv = _parse_float(row.get("ttc"))
            if tv is not None and abs(tv - 999.0) > 1e-6 and tv < 120 and tv > 0:
                ttcs.append(tv)

            ea = _parse_float(row.get("ego_a_long"))
            if ea is None:
                ea = _parse_float(row.get("ego_acceleration"))
            if ea is not None:
                acc_abs.append(abs(ea))

            ev = _parse_float(row.get("ego_v_long"))
            if ev is None:
                ev = _parse_float(row.get("ego_speed"))
            if ev is not None:
                speeds.append(ev)

    left_frac = float(left_n) / float(total) if total else 0.0
    y_range = (max(ys) - min(ys)) if len(ys) > 1 else 0.0
    v_lat_mean = _mean(v_lats) if v_lats else _mean(y_rates)
    min_thw = min(thws) if thws else 999.0
    min_ttc = min(ttcs) if ttcs else 999.0
    acc_p95 = _percentile(acc_abs, 95) if acc_abs else 0.0
    speed_p90 = _percentile(speeds, 90) if speeds else 0.0

    return {
        "n_rows": total,
        "n_segments": len(paths),
        "left_lane_fraction": left_frac,
        "y_range": y_range,
        "v_lat_abs_mean": v_lat_mean,
        "min_thw": min_thw,
        "min_ttc": min_ttc,
        "acc_abs_p95": acc_p95,
        "ego_speed_p90": speed_p90,
    }


CLUSTER_FEATURES = [
    "left_lane_fraction",
    "y_range",
    "v_lat_abs_mean",
    "min_thw",
    "min_ttc",
    "acc_abs_p95",
    "ego_speed_p90",
]


def parse_cluster_dim_weights(s):
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
    cluster_features = list(CLUSTER_FEATURES)
    if len(rows) < 3:
        raise RuntimeError("Need at least 3 drivers for k=3 clustering.")

    points = _zscore_matrix(rows, cluster_features)
    labels, _ = _kmeans(points, 3, seed, dim_weights=dim_weights)

    # Higher score => more "aggressive" overtaking style
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
                1.2 * f["left_lane_fraction"]
                + 0.9 * f["y_range"]
                + 1.0 * f["v_lat_abs_mean"]
                - 1.0 * f["min_thw"]
                - 0.8 * f["min_ttc"]
                + 0.7 * f["acc_abs_p95"]
                + 0.5 * f["ego_speed_p90"]
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
    import numpy as np

    w = np.ones(len(cluster_features), dtype=np.float64)
    if dim_weights is not None:
        w = np.asarray(dim_weights, dtype=np.float64)
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
        print("[WARN] matplotlib not installed; skip figure.")
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
    ax.set_title("Overtaking style clusters (PCA on z-scored maneuver features)")
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
        default=os.path.join(_REPO_ROOT, "overtaking", "outputs", "overtaking_il_clean_gap04"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO_ROOT, "overtaking", "outputs", "overtaking_style_clusters"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_path", type=str, default="")
    ap.add_argument("--left_y_min", type=float, default=-5.55)
    ap.add_argument("--left_y_max", type=float, default=-2.20)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.95)
    ap.add_argument(
        "--cluster_dim_weights",
        type=str,
        default="1,1,1,1,1,1,1",
        help="Weights for z-scored features, order: {}".format(",".join(CLUSTER_FEATURES)),
    )
    args = ap.parse_args()

    paths = _discover_segments(args.data_dir)
    by_driver = {}
    for p in paths:
        by_driver.setdefault(_driver_id(p), []).append(p)

    rows = []
    for driver in sorted(
        by_driver.keys(),
        key=lambda x: int(x[1:]) if x.startswith("T") and x[1:].isdigit() else 999,
    ):
        metrics = _summarize_driver_overtaking(
            by_driver[driver],
            args.left_y_min,
            args.left_y_max,
            args.right_y_min,
            args.right_y_max,
        )
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
    out_fp = os.path.join(args.out_dir, "driver_overtaking_style_clusters.csv")
    fieldnames = ["driver_id", "style_label", "cluster_id", "n_segments", "n_rows"] + cluster_features
    with open(out_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    proto_path = os.path.join(args.out_dir, "overtaking_style_prototypes.json")
    proto_summary = _style_prototype_summary(
        rows, points, cluster_features, dim_weights=dim_weights
    )
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cluster_features_zscored_pca_order": cluster_features,
                "cluster_dim_weights": dict(zip(cluster_features, dim_weights)),
                "lane_bounds": {
                    "left_y_min": args.left_y_min,
                    "left_y_max": args.left_y_max,
                    "right_y_min": args.right_y_min,
                    "right_y_max": args.right_y_max,
                },
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
            plot_fp = os.path.join(args.out_dir, "overtaking_style_clusters_pca.png")
        _save_cluster_plot(rows, points, plot_fp)


if __name__ == "__main__":
    main()
