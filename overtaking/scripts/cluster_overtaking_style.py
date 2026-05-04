# -*- coding: utf-8 -*-
"""
Cluster per-driver overtaking styles (conservative / neutral / aggressive).

Each **maneuver** is either a single ``segment_<n>.csv`` (IL 整段) or a **merged** clip:
``<run_id>__phase_01_*.csv`` … ``__phase_04_*.csv`` under the same folder (时间戳连续，按相位序号拼接).

若目录中包含 ``exp1`` / ``exp2`` / ``exp3``（独立文件夹名）或路径段 ``*_expN_*``，
脚本会 **按前车速度实验拆分**，在 ``--out_dir`` 下分别生成 ``exp1/`` ``exp2/`` ``exp3/`` 三套聚类结果；
无前述标记时仍为单次全量聚类。

Example::

  python3 overtaking/scripts/cluster_overtaking_style.py \
    --data_dir overtaking/outputs/overtaking_phase_segments_selected \
    --out_dir overtaking/outputs/overtaking_style_clusters_selected \
    --plot --seed 42
"""
from __future__ import print_function

import argparse
import csv
from collections import defaultdict
import importlib.util
import json
import math
import os
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CFS_PATH = os.path.join(_REPO_ROOT, "following", "scripts", "cluster_following_style.py")
_SOP_PATH = os.path.join(_SCRIPT_DIR, "segment_overtaking_phases.py")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfs = _load_module("cluster_following_style", _CFS_PATH)
_sop = _load_module("segment_overtaking_phases", _SOP_PATH)

_parse_float = _cfs._parse_float
_mean = _cfs._mean
_percentile = _cfs._percentile
_zscore_matrix = _cfs._zscore_matrix
_kmeans = _cfs._kmeans
_pca2 = _cfs._pca2

# english key, 说明, 风格关联预期
CLUSTER_FEATURE_ROWS = (
    (
        "ttc_init",
        "开始变道抽出瞬间的 TTC（由车间距与前车纵向速度差按采集端公式直接计算：distance_headway/(ego_speed-lead_speed)，仅在追及闭合时定义）",
        "越小越激进",
    ),
    ("dist_init", "开始变道抽出瞬间与前车的绝对距离", "越小越激进"),
    ("a_lat_max", "变道过程中的最大横向加速度 (m/s2)", "越大越激进"),
    ("jerk_lat_max", "变道过程中的最大横向冲击度 (Jerk, m/s3)", "越大越激进"),
    ("v_diff_pass", "与前车并行时的平均超车速度差 (Δv)", "越大越激进"),
    ("t_duration", "超车总耗时（从开始抽出到完全回到原车道）", "越短越激进"),
)

CLUSTER_FEATURES = [r[0] for r in CLUSTER_FEATURE_ROWS]


def feature_catalog_dictionary():
    out = []
    for key, desc, hint in CLUSTER_FEATURE_ROWS:
        out.append(
            {
                "key": key,
                "特征名": key,
                "说明": desc,
                "风格关联预期": hint,
            }
        )
    return out


def _discover_maneuvers(data_dir):
    """
    Return a list of maneuver dicts::

        { "canonical_path": str, "csv_paths": [ ordered csv ... ], "kind": "segment_il"|"phase_merge" }

    - ``segment_<n>.csv``: one maneuver per file (only if that folder has **no** ``*__phase_*`` exports).
    - ``<stem>__phase_MM_*.csv``: one maneuver per ``stem`` per folder; CSVs merged in phase order.
    """
    phase_re = re.compile(r"^(.+?)__phase_(\d+)_.+\.csv$", re.IGNORECASE)
    seg_re = re.compile(r"^segment_\d+\.csv$")
    maneuvers = []
    phase_groups = defaultdict(list)

    for root, _, files in os.walk(data_dir):
        has_phase = any(phase_re.match(fn) for fn in files)
        for fn in files:
            if not fn.endswith(".csv") or fn == "phase_segments_summary.csv":
                continue
            full = os.path.join(root, fn)
            mo = phase_re.match(fn)
            if mo:
                stem = mo.group(1)
                phase_groups[(root, stem)].append((int(mo.group(2)), full))
                continue
            if seg_re.match(fn) and not has_phase:
                maneuvers.append(
                    {
                        "canonical_path": full,
                        "csv_paths": [full],
                        "kind": "segment_il",
                    },
                )

    for key in sorted(phase_groups.keys()):
        parts = sorted(phase_groups[key], key=lambda x: x[0])
        paths_m = [p for _, p in parts]
        maneuvers.append(
            {
                "canonical_path": paths_m[0],
                "csv_paths": paths_m,
                "kind": "phase_merge",
            },
        )

    maneuvers.sort(key=lambda m: m["canonical_path"])
    return maneuvers


def _experiment_tag(segment_path):
    """
    Infer lead-speed experiment bucket from nested path segments.

    Recognizes literal folders ``exp1``, ``exp2``, ``exp3``, prefixes ``expN_`` …,
    or ``*_expN_*`` in a path component (e.g. clip folder ``*_exp2_o``).
    """
    ap = os.path.abspath(segment_path).replace("\\", "/")
    for part in ap.split("/"):
        if part in ("exp1", "exp2", "exp3"):
            return part
        mo = re.match(r"^(exp[123])(?:_|$|\.)", part)
        if mo:
            return mo.group(1)
        mo = re.search(r"_exp([123])(?:_|$|[.])", part)
        if mo:
            return "exp" + mo.group(1)
    return "_default"


def _sorted_experiment_keys(keys):
    order = {"exp1": 0, "exp2": 1, "exp3": 2}

    def _key(k):
        return (order.get(k, 10), str(k))

    return sorted(keys, key=_key)


def _group_maneuvers_by_experiment(maneuvers):
    g = defaultdict(list)
    for m in maneuvers:
        g[_experiment_tag(m["canonical_path"])].append(m)
    return g


def _use_nested_experiment_outdirs(groups):
    """Separate subdirs when any path matched ``exp[123]`` (not only pooled ``_default``)."""
    return any(k != "_default" for k in groups.keys())


def _driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T-?\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _driver_sort_key(did):
    if did == "UNKNOWN":
        return (2, 0, did)
    m = re.match(r"^T-?(\d+)$", did)
    if m:
        return (0, int(m.group(1)), "")
    return (1, 0, did)


def _semantic_style_labels(k_eff, ordered_labs_from_low_agg):
    """
    Map numeric cluster ids to names: low aggregate score → conservative, high → aggressive.
    ``ordered_labs_from_low_agg`` lists cluster ids from lowest to highest aggression proxy.
    """
    if k_eff == 1:
        return {ordered_labs_from_low_agg[0]: "neutral"}
    if k_eff == 2:
        return {
            ordered_labs_from_low_agg[0]: "conservative",
            ordered_labs_from_low_agg[1]: "aggressive",
        }
    if k_eff == 3:
        return {
            ordered_labs_from_low_agg[0]: "conservative",
            ordered_labs_from_low_agg[1]: "neutral",
            ordered_labs_from_low_agg[2]: "aggressive",
        }
    out = {}
    for rank, lab in enumerate(ordered_labs_from_low_agg):
        out[lab] = "tier_{}".format(rank)
    return out


def _ttc_usable(tv):
    if tv is None:
        return False
    if abs(tv - 999.0) <= 1e-3:
        return False
    return 0.0 < tv < 120.0


def _headway_distance_usable(dh):
    if dh is None:
        return False
    return 1e-3 < dh < 500.0


def _compute_longitudinal_closing_ttc(distance_m, ego_minus_lead_mps):
    """
    Longitudinal closing TTC: ``distance_headway / (ego_speed − lead_speed)`` when shrinking the gap.

    Matches ``replay/experiment.DataCollector`` (rear-end kinematics): ``relative_speed =
    ego_speed − lead_speed``; TTC defined only while ``relative_speed > 0.01`` m/s.

    Uses **computed** values only (do not trust CSV ``ttc`` column here).
    """
    if distance_m is None or ego_minus_lead_mps is None:
        return None
    dh = float(distance_m)
    rs = float(ego_minus_lead_mps)
    if not math.isfinite(dh) or not math.isfinite(rs):
        return None
    if rs > 1e-2:
        t = dh / rs
        if not math.isfinite(t) or t <= 0.0:
            return None
        return min(float(t), 999.0)
    return None


def _sorted_sample_rows(rows):
    acc = []
    for row in rows:
        ts = _parse_float(row.get("timestamp"))
        y = _parse_float(row.get("ego_pos_y"))
        if ts is None or y is None:
            continue
        acc.append((ts, row))
    acc.sort(key=lambda z: z[0])
    return [z[1] for z in acc]


def _central_derivative(times, vals):
    n = len(times)
    if n == 0:
        return []
    out = [0.0] * n
    if n >= 3:
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 1e-9:
                out[i] = (vals[i + 1] - vals[i - 1]) / dt
    if n >= 2:
        dt = times[1] - times[0]
        if dt > 1e-9:
            out[0] = (vals[1] - vals[0]) / dt
        dt = times[-1] - times[-2]
        if dt > 1e-9:
            out[-1] = (vals[-1] - vals[-2]) / dt
    return out


def _features_from_row_dicts(
    rows_in,
    *,
    right_y_min,
    right_y_max,
    ego_half_width_m,
    segment_smooth_window,
    edge_eps_m,
    center_hold_sec,
    y_thr_p2,
):
    """Compute feature dict from raw csv.DictReader-like row dicts (may span merged phase files).

    ``ttc_init`` uses kinematic TTC from ``distance_headway`` and longitudinal ``ego_speed−lead_speed``
    (same rule as replay data collection), never the CSV ``ttc`` field.
    """
    rows = _sorted_sample_rows(rows_in)
    n = len(rows)
    if n < 5:
        return None, 0

    times = []
    ys = []
    dhs = []
    ttc_comp = []
    rel_sp_raw = []

    for row in rows:
        times.append(float(_parse_float(row.get("timestamp"))))
        ys.append(float(_parse_float(row.get("ego_pos_y"))))

        dh_row = _parse_float(row.get("distance_headway"))
        dhs.append(dh_row)

        rs = _parse_float(row.get("relative_speed"))
        if rs is None:
            ego = _parse_float(row.get("ego_speed"))
            if ego is None:
                ego = _parse_float(row.get("ego_v_long"))
            lv = _parse_float(row.get("lead_speed"))
            if ego is not None and lv is not None:
                rs = ego - lv
        rel_sp_raw.append(rs)
        ttc_comp.append(_compute_longitudinal_closing_ttc(dh_row, rs))

    ys_seg = (
        _sop._moving_median(list(ys), int(segment_smooth_window))
        if int(segment_smooth_window) > 1
        else list(ys)
    )
    sg = _sop.segment_indices(
        times,
        ys_seg,
        float(right_y_min),
        float(right_y_max),
        float(y_thr_p2),
        float(ego_half_width_m),
        segment_smooth_window=1,
        edge_eps_m=float(edge_eps_m),
        center_hold_sec=float(center_hold_sec),
    )

    ie = int(sg["i_follow_end"])
    ir_raw = sg.get("i_reach")
    ile = int(sg["i_left_end"])

    if ir_raw is None or ie < 0 or ie >= n:
        return None, n

    ir_idx = int(ir_raw)
    ile_h = min(ile, n)

    tt0 = ttc_comp[ie]
    if not _ttc_usable(tt0):
        tt0 = None
        for j in range(ie, min(ie + 6, n)):
            if _ttc_usable(ttc_comp[j]):
                tt0 = ttc_comp[j]
                break
    if tt0 is None:
        return None, n

    d0 = dhs[ie]
    if not _headway_distance_usable(d0):
        for j in range(ie, min(ie + 6, n)):
            if _headway_distance_usable(dhs[j]):
                d0 = dhs[j]
                break
    if not _headway_distance_usable(d0):
        return None, n

    vy = _central_derivative(times, ys)
    ay = _central_derivative(times, vy)
    jy = _central_derivative(times, ay)

    lo_lc = ie
    hi_lc = max(ie + 1, ile_h)
    peak_a = 0.0
    peak_j = 0.0
    for kk in range(lo_lc, hi_lc):
        peak_a = max(peak_a, abs(ay[kk]))
        peak_j = max(peak_j, abs(jy[kk]))

    v_samples = []
    for kk in range(ir_idx, min(ile_h, n)):
        rs = rel_sp_raw[kk]
        if rs is not None and math.isfinite(float(rs)):
            v_samples.append(float(rs))
    if not v_samples:
        return None, n

    v_diff_mean = float(_mean(v_samples))
    if ir_idx >= ile_h:
        return None, n

    t_duration = float(times[-1]) - float(times[ie])

    return (
        {
            "ttc_init": float(tt0),
            "dist_init": float(d0),
            "a_lat_max": float(peak_a),
            "jerk_lat_max": float(peak_j),
            "v_diff_pass": v_diff_mean,
            "t_duration": t_duration,
        },
        n,
    )


def _features_maneuver(
    maneuver,
    *,
    right_y_min,
    right_y_max,
    ego_half_width_m,
    segment_smooth_window,
    edge_eps_m,
    center_hold_sec,
    y_thr_p2,
):
    rows_in = []
    for fp in maneuver["csv_paths"]:
        with open(fp, "r", encoding="utf-8") as f:
            rows_in.extend(list(csv.DictReader(f)))
    return _features_from_row_dicts(
        rows_in,
        right_y_min=right_y_min,
        right_y_max=right_y_max,
        ego_half_width_m=ego_half_width_m,
        segment_smooth_window=segment_smooth_window,
        edge_eps_m=edge_eps_m,
        center_hold_sec=center_hold_sec,
        y_thr_p2=y_thr_p2,
    )




def _median_list(vals):
    if not vals:
        return 0.0
    return float(_percentile(vals, 50))


def summarize_driver_maneuvers(
    maneuvers,
    *,
    right_y_min,
    right_y_max,
    ego_half_width_m,
    segment_smooth_window,
    edge_eps_m,
    center_hold_sec,
    y_center_p2_end,
):
    feats_acc = {
        k: []
        for k in CLUSTER_FEATURES
    }
    n_rows = 0
    n_seg_ok = 0

    for m in maneuvers:
        rec, n_r = _features_maneuver(
            m,
            right_y_min=right_y_min,
            right_y_max=right_y_max,
            ego_half_width_m=ego_half_width_m,
            segment_smooth_window=segment_smooth_window,
            edge_eps_m=edge_eps_m,
            center_hold_sec=center_hold_sec,
            y_thr_p2=y_center_p2_end,
        )
        n_rows += n_r
        if rec is None:
            continue
        n_seg_ok += 1
        for k in CLUSTER_FEATURES:
            feats_acc[k].append(rec[k])

    if n_seg_ok == 0:
        return None

    return {
        "n_rows": n_rows,
        "n_segments": len(maneuvers),
        "n_segments_valid": n_seg_ok,
        **{k: _median_list(feats_acc[k]) for k in CLUSTER_FEATURES},
    }


def parse_cluster_dim_weights(s):
    dim_weights = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
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


def _point_aggression_proxy(cluster_features, vec):
    p = dict(zip(cluster_features, vec))
    return (
        -p["ttc_init"]
        - p["dist_init"]
        + p["a_lat_max"]
        + p["jerk_lat_max"]
        + p["v_diff_pass"]
        - p["t_duration"]
    )


def assign_kmeans_styles(rows, dim_weights, seed, n_clusters=3):
    cluster_features = list(CLUSTER_FEATURES)
    n_d = len(rows)
    if n_d == 0:
        raise RuntimeError(
            "No drivers left after feature extraction. "
            "Found no usable maneuvers (segment_*.csv or merged *__phase_* exports) with valid "
            "computed TTC / headway / phase metrics. "
            "Check --data_dir tree, CSV columns (timestamp, ego_pos_y, ego_speed, lead_speed, "
            "distance_headway), and lane / P2 threshold flags (--right_y_min/max, --y_center_p2_end, --p2_end_mode)."
        )

    k_req = max(1, int(n_clusters))
    k_eff = min(k_req, n_d)
    if k_eff < k_req:
        print(
            "[INFO] k-means: requested k={} but only {} driver(s); using k={}".format(
                k_req, n_d, k_eff
            )
        )

    points = _zscore_matrix(rows, cluster_features)
    labels, _ = _kmeans(points, k_eff, seed, dim_weights=dim_weights)

    cluster_scores = {}
    for lab in range(k_eff):
        idxs = [i for i, x in enumerate(labels) if x == lab]
        if not idxs:
            cluster_scores[lab] = 0.0
            continue
        score_sum = sum(_point_aggression_proxy(cluster_features, points[i]) for i in idxs)
        cluster_scores[lab] = score_sum / float(len(idxs))

    ordered = sorted(range(k_eff), key=lambda x: cluster_scores[x])
    label_name = _semantic_style_labels(k_eff, ordered)

    for i, r in enumerate(rows):
        r["cluster_id"] = labels[i]
        r["style_label"] = label_name[labels[i]]

    return {
        "points": points,
        "cluster_features": cluster_features,
        "numeric_cluster_to_style": label_name,
        "k_requested": k_req,
        "k_effective": k_eff,
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


def run_clustering_bundle(
    bundle_maneuvers,
    experiment_tag,
    args,
    y_thr,
    dest_dir,
    *,
    plot_path_override,
):
    """
    Build features, cluster, write CSV/JSON/optional plot under ``dest_dir``.

    ``plot_path_override``: if non-empty, use as PNG path when ``--plot``; else default under dest_dir.
    Returns a JSON-serializable summary dict.
    """
    tag_log = experiment_tag if experiment_tag != "_merged" else "all_maneuvers"
    n_csv_files = sum(len(m["csv_paths"]) for m in bundle_maneuvers)
    print(
        "[INFO] --- bundle {!r} --- maneuvers: {} ({} CSV files)".format(
            tag_log,
            len(bundle_maneuvers),
            n_csv_files,
        ),
    )

    by_driver = defaultdict(list)
    for mm in bundle_maneuvers:
        by_driver[_driver_id(mm["canonical_path"])].append(mm)
    print("[INFO] bundle {!r}: distinct driver buckets: {}".format(tag_log, len(by_driver)))

    rows = []
    for driver in sorted(by_driver.keys(), key=_driver_sort_key):
        m = summarize_driver_maneuvers(
            by_driver[driver],
            right_y_min=args.right_y_min,
            right_y_max=args.right_y_max,
            ego_half_width_m=args.ego_half_width_m,
            segment_smooth_window=args.segment_smooth_window,
            edge_eps_m=args.edge_eps_m,
            center_hold_sec=args.center_hold_sec,
            y_center_p2_end=y_thr,
        )
        if m is None:
            print("[WARN] bundle {!r}: skip driver (no valid segments): {}".format(tag_log, driver))
            continue
        m["driver_id"] = driver
        rows.append(m)

    rec = {
        "experiment_tag": experiment_tag,
        "log_label": tag_log,
        "output_dir": os.path.abspath(dest_dir),
        "maneuvers_input": len(bundle_maneuvers),
        "csv_files_total": n_csv_files,
        "distinct_driver_buckets": len(by_driver),
        "drivers_with_features": len(rows),
        "ok": False,
    }

    if not rows:
        print(
            "[WARN] bundle {!r}: no drivers with valid features — skip clustering & files.".format(
                tag_log,
            ),
        )
        rec["ok"] = False
        rec["skip_reason"] = "no_valid_features"
        return rec

    cluster_features = list(CLUSTER_FEATURES)
    dim_weights = parse_cluster_dim_weights(args.cluster_dim_weights)
    print(
        "[INFO] bundle {!r}: k-means dim weights: {}".format(
            tag_log,
            ", ".join(
                "{}={}".format(nm, wt) for nm, wt in zip(cluster_features, dim_weights)
            ),
        ),
    )
    print("[INFO] bundle {!r}: drivers used for clustering: {}".format(tag_log, len(rows)))

    result = assign_kmeans_styles(rows, dim_weights, args.seed, n_clusters=args.n_clusters)
    points = result["points"]

    os.makedirs(dest_dir, exist_ok=True)
    out_fp = os.path.join(dest_dir, "driver_overtaking_style_clusters.csv")
    meta_cols = ["n_segments", "n_segments_valid", "n_rows"]
    fieldnames = ["driver_id", "style_label", "cluster_id"] + meta_cols + cluster_features
    with open(out_fp, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=fieldnames)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r.get(k, "") for k in fieldnames})

    proto_summary = _style_prototype_summary(
        rows, points, cluster_features, dim_weights=dim_weights,
    )

    fc_path = os.path.join(dest_dir, "overtaking_feature_catalog.json")
    cat = feature_catalog_dictionary()
    with open(fc_path, "w", encoding="utf-8") as f_fc:
        json.dump(cat, f_fc, ensure_ascii=False, indent=2)

    proto_path = os.path.join(dest_dir, "overtaking_style_prototypes.json")
    with open(proto_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_tag": experiment_tag,
                "feature_catalog_file": os.path.basename(fc_path),
                "feature_catalog_embedded": cat,
                "cluster_features_zscored_pca_order": cluster_features,
                "cluster_dim_weights": dict(zip(cluster_features, dim_weights)),
                "clustering_k_requested": result.get("k_requested"),
                "clustering_k_effective": result.get("k_effective"),
                "numeric_cluster_to_style": result.get("numeric_cluster_to_style"),
                "segment_source": (
                    "segment_overtaking_phases.segment_indices on time-ordered rows; "
                    "phase_merge = concat *__phase_MM_*.csv per clip; "
                    "ttc_init = distance_headway/(ego_speed-lead_speed) when (ego_speed-lead_speed)>1e-2 "
                    "(replay/DataCollector kinematics; not CSV ttc column); "
                    "t_duration = timestamp[last] - timestamp[i_follow_end]; "
                    "a_lat/jerk from central differences on ego_pos_y"
                ),
                "lane_bounds": {
                    "left_y_min": args.left_y_min,
                    "left_y_max": args.left_y_max,
                    "right_y_min": args.right_y_min,
                    "right_y_max": args.right_y_max,
                },
                "p2_threshold": {"mode": args.p2_end_mode, "y_thr": y_thr},
                "styles": proto_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[OK] bundle {!r}: feature catalog: {}".format(tag_log, fc_path))
    print("[OK] bundle {!r}: prototypes: {}".format(tag_log, proto_path))
    print("[OK] bundle {!r}: drivers: {}".format(tag_log, len(rows)))
    print("[OK] bundle {!r}: output: {}".format(tag_log, out_fp))
    for r in rows:
        print("  {} / {}: {}".format(tag_log, r["driver_id"], r["style_label"]))

    subtitle = "" if experiment_tag == "_merged" else str(experiment_tag)
    if args.plot:
        if plot_path_override:
            plot_fp = plot_path_override
        else:
            plot_fp = os.path.join(dest_dir, "overtaking_style_clusters_pca.png")
        _save_cluster_plot(rows, points, plot_fp, subtitle=subtitle)

    rec["ok"] = True
    rec["csv_path"] = out_fp
    rec["prototypes_path"] = proto_path
    rec["feature_catalog_path"] = fc_path
    rec["clustering_k_effective"] = result.get("k_effective")
    return rec


def _save_cluster_plot(rows, zpoints, out_path, subtitle=""):
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

    palette = [
        "#27ae60",
        "#3498db",
        "#e74c3c",
        "#9b59b6",
        "#f39c12",
        "#1abc9c",
        "#34495e",
        "#e67e22",
    ]
    styles_present = []
    seen = set()
    for r in rows:
        s = r.get("style_label")
        if s not in seen:
            seen.add(s)
            styles_present.append(s)

    fig, ax = plt.subplots(figsize=(9, 7))
    for si, style in enumerate(styles_present):
        idx = [i for i, r in enumerate(rows) if r.get("style_label") == style]
        if not idx:
            continue
        c = palette[si % len(palette)]
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=120,
            c=c,
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
    ax.set_title("Overtaking style clusters (PCA on z-scored maneuver features){}".format(
        " [{}]".format(subtitle) if subtitle else ""
    ))
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
    ap.add_argument("--ego_half_width_m", type=float, default=0.9)
    ap.add_argument("--p2_end_mode", choices=("fixed", "geometry"), default="fixed")
    ap.add_argument("--y_center_p2_end", type=float, default=None)
    ap.add_argument("--segment_smooth_window", type=int, default=5)
    ap.add_argument("--edge_eps_m", type=float, default=0.08)
    ap.add_argument("--center_hold_sec", type=float, default=0.15)
    ap.add_argument(
        "--cluster_dim_weights",
        type=str,
        default="1,1,1,1,1,1",
        help="Weights for z-scored features (order {})".format(",".join(CLUSTER_FEATURES)),
    )
    ap.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="k-means cluster count (capped at number of drivers with valid features; default 3).",
    )
    ap.add_argument(
        "--no_split_by_experiment",
        action="store_true",
        help=(
            "Ignore exp1/exp2/exp3 path grouping; merge all segments into one clustering "
            "(single output folder)."
        ),
    )
    args = ap.parse_args()

    y_thr = (
        float(args.right_y_max) + float(args.ego_half_width_m)
        if args.p2_end_mode == "geometry"
        else (
            float(args.y_center_p2_end)
            if args.y_center_p2_end is not None
            else -4.5
        )
    )

    maneuvers = _discover_maneuvers(args.data_dir)
    n_il = sum(1 for m in maneuvers if m["kind"] == "segment_il")
    n_pm = sum(1 for m in maneuvers if m["kind"] == "phase_merge")
    print(
        "[INFO] maneuvers (clips): {}  (segment_il={}, phase_merge={})".format(
            len(maneuvers),
            n_il,
            n_pm,
        ),
    )
    if not maneuvers:
        print(
            "[WARN] No inputs found under {}; need segment_<n>.csv or *__phase_*_*.csv. Exit.".format(
                args.data_dir,
            ),
        )
        os.makedirs(args.out_dir, exist_ok=True)
        sum_path = os.path.join(
            os.path.abspath(args.out_dir),
            "clustering_bundle_summary.json",
        )
        with open(sum_path, "w", encoding="utf-8") as fs:
            json.dump(
                {
                    "data_dir": os.path.abspath(args.data_dir),
                    "error": "no_maneuver_csv_found",
                    "bundles": [],
                },
                fs,
                ensure_ascii=False,
                indent=2,
            )
        print("[OK] wrote empty summary:", sum_path)
        return

    groups = _group_maneuvers_by_experiment(maneuvers)
    for ek in _sorted_experiment_keys(list(groups.keys())):
        gm = groups[ek]
        n_csv = sum(len(m["csv_paths"]) for m in gm)
        print(
            "[INFO] path tag {!r}: {} maneuver(s), {} CSV file(s)".format(ek, len(gm), n_csv),
        )

    nested_layout = False if args.no_split_by_experiment else _use_nested_experiment_outdirs(groups)

    bundles = []
    if nested_layout:
        for ek in _sorted_experiment_keys(list(groups.keys())):
            mlist = groups[ek]
            if not mlist:
                continue
            dest = os.path.join(args.out_dir, ek)
            bundles.append((ek, mlist, dest))
        layout_note = (
            "per-experiment outputs under {}; tags from path (exp1/exp2/exp3 or *_expN_*)".format(
                os.path.abspath(args.out_dir),
            )
        )
    else:
        dest = os.path.abspath(args.out_dir)
        bundles.append(("_merged", maneuvers, dest))
        layout_note = "single merged output folder"

    print("[INFO] layout: {} ({} bundles)".format(layout_note, len(bundles)))

    if not bundles:
        print("[WARN] No bundles to cluster; exit.")
        return

    multi_bundle = len(bundles) > 1

    bundle_summaries = []
    os.makedirs(args.out_dir, exist_ok=True)

    pp = (
        os.path.abspath(args.plot_path.strip())
        if (
            multi_bundle is False
            and args.plot
            and isinstance(args.plot_path, str)
            and args.plot_path.strip()
        )
        else ""
    )

    for exp_tag, bundle_maneuvers, dest_dir in bundles:
        bundle_summaries.append(
            run_clustering_bundle(
                bundle_maneuvers,
                exp_tag,
                args,
                y_thr,
                dest_dir,
                plot_path_override=pp,
            ),
        )
        if pp:
            pp = ""

    sum_path = os.path.join(os.path.abspath(args.out_dir), "clustering_bundle_summary.json")
    with open(sum_path, "w", encoding="utf-8") as fs:
        json.dump(
            {
                "data_dir": os.path.abspath(args.data_dir),
                "nested_experiment_dirs": nested_layout,
                "no_split_by_experiment_flag": args.no_split_by_experiment,
                "experiment_maneuver_counts_pre_split": {k: len(v) for k, v in groups.items()},
                "bundles": bundle_summaries,
            },
            fs,
            ensure_ascii=False,
            indent=2,
        )
    print("[OK] run summary:", sum_path)


if __name__ == "__main__":
    main()
