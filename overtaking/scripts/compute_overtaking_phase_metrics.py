# -*- coding: utf-8 -*-
"""
Per-trial overtaking metrics from ``phase_segments_summary.csv`` + raw ``driving_data.csv``.

Reads summary columns: ``file``, ``status``, ``i_end_following`` (P1|P2 boundary),
``i_p2_end`` (lane-change end / P3 start), ``i_end_left_overtake`` (P4 start).

Phases (row indices, half-open where noted):
  P2 lane change: ``[i_end_following, i_p2_end)``
  P3 left overtake: ``[i_p2_end, i_end_left_overtake)``
  P4 return: ``[i_end_left_overtake, n)``

Metrics (per experiment / session CSV):
  • P2 **onset**: ``distance_headway`` at index ``i_end_following`` (first P2 sample).
  • P2 longitudinal accel: stats on ``ego_acceleration`` during P2.
  • P2 lateral accel: from ``ego_pos_y`` via finite-diff velocity then accel (same time base).
  • P2 minimum TTC over valid samples (excludes sentinel ~999).
  • P3 **lateral excursion**: ``max(ego_pos_y) - min(ego_pos_y)`` on P3.
  • P3 longitudinal accel: stats on ``ego_acceleration``.
  • P4 **onset**: ``distance_headway`` at ``i_end_left_overtake`` (first P4 sample).
  • P4 minimum TTC (valid samples only).

Quantiles default: p10, p25, p50 (median), p75, p90.

``--y_smooth_window`` (default 5): odd moving median on ``ego_pos_y`` **only** for lateral-accel
finite-differencing; use 1 to disable.

Example::

  python3 overtaking/scripts/compute_overtaking_phase_metrics.py \\
    --summary_csv overtaking/outputs/overtaking_phase_segments/phase_segments_summary.csv \\
    --data_root data \\
    --out_csv overtaking/outputs/overtaking_phase_metrics.csv
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import re


def _parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_valid_ttc(v):
    if v is None:
        return False
    if math.isnan(v) or math.isinf(v):
        return False
    if v >= 998.0:
        return False
    if v <= 0.0:
        return False
    return True


def _pctile(vals, p):
    """Linear interpolation percentile in [0,1], vals non-empty."""
    if not vals:
        return None
    x = sorted(vals)
    n = len(x)
    if n == 1:
        return x[0]
    k = (n - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return x[f]
    return x[f] * (c - k) + x[c] * (k - f)


def _stats(vals, probs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    out = {}
    if not vals:
        for pq in probs:
            out[int(round(pq * 100))] = None
        return out
    for pq in probs:
        key = int(round(pq * 100))
        out[key] = _pctile(vals, pq)
    return out


def _format_stats(prefix, stats_map):
    row = {}
    for k, v in stats_map.items():
        row["%s_p%02d" % (prefix, k)] = "" if v is None else "{:.6f}".format(v)
    return row


def _moving_median(arr, window):
    if window <= 1 or not arr:
        return list(arr)
    w = int(window) | 1
    half = w // 2
    out = []
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        chunk = sorted(arr[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def lateral_accel_series(times, ys):
    """Per-index lateral acceleration (y); None where undefined. Same length as input."""
    n = len(times)
    out = [None] * n
    if n < 3:
        return out
    vy = [None] * n
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt > 1e-9:
            vy[i] = (ys[i] - ys[i - 1]) / dt
    for i in range(1, n - 1):
        if vy[i] is None or vy[i + 1] is None:
            continue
        dt = times[i + 1] - times[i]
        if dt > 1e-9:
            out[i] = (vy[i + 1] - vy[i]) / dt
    return out



def analyze_trial(rows, i_follow, i_p2_end, i_p4_start, y_smooth_window=5):
    """
    i_follow = i_end_following (P2 start idx)
    i_p2_end = exclusive end P2 / P3 start
    i_p4_start = i_end_left_overtake
    """
    n = len(rows)
    probs = (0.1, 0.25, 0.5, 0.75, 0.9)

    times = [_parse_float(r.get("timestamp")) for r in rows]
    ys = [_parse_float(r.get("ego_pos_y")) for r in rows]

    accel_long_series = [_parse_float(r.get("ego_acceleration")) for r in rows]

    if n > 0 and all(t is not None for t in times) and all(y is not None for y in ys):
        ys_lat = _moving_median(ys, y_smooth_window) if y_smooth_window > 1 else ys
        a_lat_series = lateral_accel_series(times, ys_lat)
    else:
        a_lat_series = [None] * n

    def headway(idx):
        if idx is None or idx < 0 or idx >= n:
            return None
        return _parse_float(rows[idx].get("distance_headway"))

    def ttc_series(a, b):
        tt = []
        for i in range(max(0, a), min(n, b)):
            v = _parse_float(rows[i].get("ttc"))
            if _is_valid_ttc(v):
                tt.append(v)
        return tt

    row = {}

    row["idx_p2_start"] = str(i_follow) if i_follow is not None else ""
    row["idx_p3_start"] = str(i_p2_end) if i_p2_end is not None else ""
    row["idx_p4_start"] = str(i_p4_start) if i_p4_start is not None else ""

    dh_p2_on = headway(i_follow) if i_follow is not None else None
    row["p2_onset_distance_headway_m"] = "" if dh_p2_on is None else "{:.6f}".format(dh_p2_on)

    p2_long = []
    p2_lat = []
    if i_follow is not None and i_p2_end is not None and i_follow < i_p2_end:
        for i in range(i_follow, i_p2_end):
            v = accel_long_series[i]
            if v is not None:
                p2_long.append(v)
            w = a_lat_series[i]
            if w is not None:
                p2_lat.append(w)
        t2 = ttc_series(i_follow, i_p2_end)
        row["p2_ttc_min_s"] = "" if not t2 else "{:.6f}".format(min(t2))
        row.update(_format_stats("p2_ego_accel_long", _stats(p2_long, probs)))
        row.update(_format_stats("p2_ego_accel_lat", _stats(p2_lat, probs)))
    else:
        row["p2_ttc_min_s"] = ""
        row.update(_format_stats("p2_ego_accel_long", _stats([], probs)))
        row.update(_format_stats("p2_ego_accel_lat", _stats([], probs)))

    p3_lat_range = ""
    if i_p2_end is not None and i_p4_start is not None and i_p2_end < i_p4_start:
        seg_y = ys[i_p2_end : i_p4_start]
        seg_y_fin = [v for v in seg_y if v is not None]
        if seg_y_fin:
            p3_lat_range = "{:.6f}".format(max(seg_y_fin) - min(seg_y_fin))
    row["p3_ego_lat_range_m"] = p3_lat_range

    p3_long = []
    if i_p2_end is not None and i_p4_start is not None and i_p2_end < i_p4_start:
        for i in range(i_p2_end, i_p4_start):
            v = accel_long_series[i]
            if v is not None:
                p3_long.append(v)
    row.update(_format_stats("p3_ego_accel_long", _stats(p3_long, probs)))

    dh_p4 = headway(i_p4_start) if i_p4_start is not None else None
    row["p4_onset_distance_headway_m"] = "" if dh_p4 is None else "{:.6f}".format(dh_p4)

    t4 = ttc_series(i_p4_start, n) if i_p4_start is not None else []
    row["p4_ttc_min_s"] = "" if not t4 else "{:.6f}".format(min(t4))

    return row


def main():
    ap = argparse.ArgumentParser(description="Overtaking P2/P3/P4 descriptive metrics → CSV.")
    ap.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="phase_segments_summary.csv from segment_overtaking_phases.py",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root joining summary ``file`` column to load driving_data.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="overtaking_phase_metrics.csv",
    )
    ap.add_argument(
        "--y_smooth_window",
        type=int,
        default=5,
        help="Odd moving median window on ego_pos_y before lateral FD accel; 1 disables.",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="0 = process all rows in summary CSV.",
    )
    args = ap.parse_args()

    summary_path = args.summary_csv
    if not os.path.isfile(summary_path):
        raise SystemExit("summary not found: %s" % summary_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        sr = csv.DictReader(f)
        sums = list(sr)

    def _pick_int(row, *keys):
        for k in keys:
            if k in row and str(row[k]).strip() != "":
                try:
                    return int(float(row[k]))
                except ValueError:
                    continue
        return None

    out_rows = []
    for i_rec, summ in enumerate(sums):
        if args.max_files and i_rec >= args.max_files:
            break

        rel = (summ.get("file") or "").strip().replace("\\", "/")
        st = (summ.get("status") or "").strip()
        sess = ""
        parts = rel.split("/")
        for p in parts:
            if "exp" in p and "_" in p:
                sess = os.path.basename(p)
                break
        if not sess:
            sess = parts[-2] if len(parts) >= 2 else ""

        i_follow = _pick_int(summ, "i_end_following")
        i_p2_end = _pick_int(summ, "i_p2_end", "i_p3_end", "i_end_lane_change_exclusive")
        i_p4_start = _pick_int(summ, "i_end_left_overtake")

        mexp = re.search(r"(exp[123]_o)", rel)
        exp_tag = mexp.group(1) if mexp else ""

        fp = os.path.join(args.data_root, rel.replace("/", os.sep))
        base_row = {"file": rel, "session_id": sess, "trial_status": st, "exp_tag": exp_tag}

        if not os.path.isfile(fp):
            base_row.update(
                {
                    "_error": "missing_file",
                    "n_rows_loaded": "",
                    "p2_onset_distance_headway_m": "",
                    "p4_onset_distance_headway_m": "",
                }
            )
            out_rows.append(base_row)
            continue

        with open(fp, "r", encoding="utf-8") as fh:
            rdr = csv.DictReader(fh)
            rows = list(rdr)

        metrics = analyze_trial(rows, i_follow, i_p2_end, i_p4_start, y_smooth_window=args.y_smooth_window)
        base_row.update(metrics)
        base_row["n_rows_loaded"] = str(len(rows))
        out_rows.append(base_row)

    if not out_rows:
        raise SystemExit("no output rows")

    fixed = [
        "file",
        "session_id",
        "exp_tag",
        "trial_status",
        "n_rows_loaded",
        "idx_p2_start",
        "idx_p3_start",
        "idx_p4_start",
        "p2_onset_distance_headway_m",
        "p2_ttc_min_s",
        "p2_ego_accel_lat_p10",
        "p2_ego_accel_lat_p25",
        "p2_ego_accel_lat_p50",
        "p2_ego_accel_lat_p75",
        "p2_ego_accel_lat_p90",
        "p2_ego_accel_long_p10",
        "p2_ego_accel_long_p25",
        "p2_ego_accel_long_p50",
        "p2_ego_accel_long_p75",
        "p2_ego_accel_long_p90",
        "p3_ego_lat_range_m",
        "p3_ego_accel_long_p10",
        "p3_ego_accel_long_p25",
        "p3_ego_accel_long_p50",
        "p3_ego_accel_long_p75",
        "p3_ego_accel_long_p90",
        "p4_onset_distance_headway_m",
        "p4_ttc_min_s",
    ]

    extras = sorted(
        set().union(*(r.keys() for r in out_rows)) - set(fixed) - {"_error"}
    )
    cols = fixed + extras

    od = os.path.dirname(os.path.abspath(args.out_csv))
    if od:
        os.makedirs(od, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print("[OK] wrote", len(out_rows), "rows →", args.out_csv)


if __name__ == "__main__":
    main()
