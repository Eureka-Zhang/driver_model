# -*- coding: utf-8 -*-
"""
Batch quality scores for ``driving_data.csv`` under a directory tree.

Scores each file 0–100 plus component metrics (time base, gaps, completeness, jumps).
Optional filters: overtaking-only (``exp[123]_o``), following-only, or all.

Example::


    
    python3 overtaking/scripts/batch_score_driving_data_quality.py \
  --data_dir /home/zwx/driver_model/data/T2 \
  --out_csv overtaking/outputs/T2_driving_data_quality_scores.csv \
  --discovery overtaking \
  --gap_threshold_sec 0.45 \
  --max_speed_jump_mps 20 \
  --max_pos_jump_m 12 \
  --min_rows 80
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

ESSENTIAL_FIELDS = [
    "timestamp",
    "ego_pos_x",
    "ego_pos_y",
    "ego_speed",
    "ego_acceleration",
    "throttle",
    "brake",
    "steer",
    "lead_pos_x",
    "lead_pos_y",
    "lead_speed",
    "distance_headway",
]

REPLAY_CORE_FIELDS = [
    "timestamp",
    "ego_pos_x",
    "ego_pos_y",
    "ego_speed",
    "lead_pos_x",
    "lead_pos_y",
]


def _parse_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _discover_csvs(data_dir, mode):
    """mode: all | overtaking | following"""
    cands = []
    for root, _, files in os.walk(data_dir):
        if "driving_data.csv" not in files:
            continue
        p = root.replace("\\", "/")
        if "pre_familiarization" in p:
            continue
        if "_b" in p:
            continue
        if mode == "overtaking":
            if not re.search(r"exp[123]_o", p):
                continue
        elif mode == "following":
            if re.search(r"/[^/]*_o(?:/|$)", p):
                continue
            if not (("following" in p) or re.search(r"/[^/]*_f(?:/|$)", p)):
                continue
        cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def _grade(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def score_file(
    rows,
    fieldnames,
    gap_threshold_sec,
    max_speed_jump_mps,
    max_pos_jump_m,
    min_rows,
):
    out = {
        "n_rows": len(rows),
        "duration_sec": "",
        "missing_header_required": "0",
        "rows_missing_any_essential": "",
        "missing_essential_frac": "",
        "replay_core_missing_frac": "",
        "ts_none_count": "",
        "mono_pairs_ratio": "",
        "duplicate_ts_count": "",
        "max_dt_sec": "",
        "mean_dt_sec": "",
        "n_dt_gaps_over_thresh": "",
        "n_speed_jumps": "",
        "n_pos_jumps": "",
        "ttc_sentinel_frac": "",
        "th_sentinel_frac": "",
        "score_overall": "",
        "grade": "",
        "issues_summary": "",
    }

    req = set(ESSENTIAL_FIELDS)
    replay_req = set(REPLAY_CORE_FIELDS)
    if not fieldnames:
        out["missing_header_required"] = "1"
        out["score_overall"] = "0"
        out["grade"] = "F"
        out["issues_summary"] = "no_header"
        return out

    missing_cols = req - set(fieldnames)
    if missing_cols:
        out["missing_header_required"] = "1"
        out["score_overall"] = "15"
        out["grade"] = "F"
        out["issues_summary"] = "missing_columns:%s" % ",".join(sorted(missing_cols)[:8])
        return out

    n = len(rows)
    # Completeness on raw rows
    miss_any = 0
    miss_replay = 0
    for r in rows:
        bad = False
        for k in ESSENTIAL_FIELDS:
            if _parse_float(r.get(k)) is None:
                bad = True
                break
        if bad:
            miss_any += 1
        rb = False
        for k in REPLAY_CORE_FIELDS:
            if k == "timestamp" and _parse_float(r.get(k)) is None:
                rb = True
                break
            if k != "timestamp" and _parse_float(r.get(k)) is None:
                rb = True
                break
        if rb:
            miss_replay += 1

    out["rows_missing_any_essential"] = str(miss_any)
    out["missing_essential_frac"] = "{:.4f}".format(miss_any / max(1, n))
    out["replay_core_missing_frac"] = "{:.4f}".format(miss_replay / max(1, n))

    ts_list = [_parse_float(r.get("timestamp")) for r in rows]
    ts_none = sum(1 for t in ts_list if t is None)
    out["ts_none_count"] = str(ts_none)

    pairs_ok = pairs_bad = 0
    for i in range(1, n):
        a, b = ts_list[i - 1], ts_list[i]
        if a is None or b is None:
            continue
        if b > a:
            pairs_ok += 1
        else:
            pairs_bad += 1
    mono_denom = pairs_ok + pairs_bad
    mono_ratio = (pairs_ok / mono_denom) if mono_denom else 1.0
    out["mono_pairs_ratio"] = "{:.4f}".format(mono_ratio)

    indexed = [(ts_list[i], i) for i in range(n)]
    indexed = [x for x in indexed if x[0] is not None]
    indexed.sort(key=lambda x: x[0])

    dup_ts = 0
    dts = []
    n_speed_jump = n_pos_jump = 0

    if len(indexed) < 2:
        out["duplicate_ts_count"] = "0"
        out["max_dt_sec"] = ""
        out["mean_dt_sec"] = ""
        out["n_dt_gaps_over_thresh"] = "0"
    else:
        for k in range(1, len(indexed)):
            t0, t1 = indexed[k - 1][0], indexed[k][0]
            if abs(t1 - t0) < 1e-12:
                dup_ts += 1
            else:
                dts.append(t1 - t0)
            if abs(t1 - t0) < 1e-12:
                continue
            i0, i1 = indexed[k - 1][1], indexed[k][1]
            r0, r1 = rows[i0], rows[i1]
            v0 = _parse_float(r0.get("ego_speed"))
            v1 = _parse_float(r1.get("ego_speed"))
            if v0 is not None and v1 is not None and abs(v1 - v0) > max_speed_jump_mps:
                n_speed_jump += 1
            ex0 = _parse_float(r0.get("ego_pos_x"))
            ey0 = _parse_float(r0.get("ego_pos_y"))
            ex1 = _parse_float(r1.get("ego_pos_x"))
            ey1 = _parse_float(r1.get("ego_pos_y"))
            if (
                ex0 is not None
                and ey0 is not None
                and ex1 is not None
                and ey1 is not None
            ):
                dj = math.hypot(ex1 - ex0, ey1 - ey0)
                if dj > max_pos_jump_m:
                    n_pos_jump += 1

        out["duplicate_ts_count"] = str(dup_ts)
        out["max_dt_sec"] = "{:.6f}".format(max(dts)) if dts else ""
        out["mean_dt_sec"] = "{:.6f}".format(sum(dts) / len(dts)) if dts else ""
        out["n_dt_gaps_over_thresh"] = str(sum(1 for dt in dts if dt > gap_threshold_sec))

    out["n_speed_jumps"] = str(n_speed_jump)
    out["n_pos_jumps"] = str(n_pos_jump)

    tsent = hsent = 0
    for r in rows:
        tv = _parse_float(r.get("ttc"))
        if tv is not None and abs(tv - 999.0) < 1e-6:
            tsent += 1
        hv = _parse_float(r.get("time_headway"))
        if hv is not None and abs(hv - 999.0) < 1e-6:
            hsent += 1

    out["ttc_sentinel_frac"] = "{:.4f}".format(tsent / max(1, n))
    out["th_sentinel_frac"] = "{:.4f}".format(hsent / max(1, n))

    if indexed:
        dur = indexed[-1][0] - indexed[0][0]
        out["duration_sec"] = "{:.6f}".format(max(0.0, dur))

    # --- Overall score 0–100 ---
    missing_frac = miss_any / max(1, n)
    n_gap = int(out["n_dt_gaps_over_thresh"])
    replay_miss_frac = miss_replay / max(1, n)

    score = 100.0
    score -= 42.0 * min(1.0, missing_frac * 2.5)
    score -= 18.0 * min(1.0, replay_miss_frac * 3.0)
    score -= 15.0 * (1.0 - mono_ratio)
    score -= min(12.0, n_gap * 2.5)
    score -= min(8.0, dup_ts * 0.25)
    jpen = min(18.0, (n_speed_jump + n_pos_jump) * (60.0 / max(80, len(indexed))))
    score -= jpen
    if n < min_rows:
        score -= 12.0
    score = max(0.0, min(100.0, score))

    out["score_overall"] = "{:.2f}".format(score)
    out["grade"] = _grade(score)

    parts = []
    if missing_frac > 0.02:
        parts.append("many_missing_essential")
    if replay_miss_frac > 0.02:
        parts.append("replay_cols_missing")
    if mono_ratio < 0.99:
        parts.append("non_monotonic_ts")
    if n_gap > 0:
        parts.append("large_time_gaps")
    if dup_ts > 0:
        parts.append("duplicate_timestamp")
    if n_speed_jump or n_pos_jump:
        parts.append("physical_jumps")
    if n < min_rows:
        parts.append("short_file")
    out["issues_summary"] = ";".join(parts) if parts else "clean"

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Batch score driving_data.csv quality (replay / IL readiness).",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(_REPO, "data"),
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join(_REPO, "overtaking", "outputs", "driving_data_quality_scores.csv"),
    )
    ap.add_argument(
        "--discovery",
        choices=("all", "overtaking", "following"),
        default="overtaking",
        help="Which driving_data.csv files to scan (default: overtaking exp*_o).",
    )
    ap.add_argument("--gap_threshold_sec", type=float, default=0.45)
    ap.add_argument(
        "--max_speed_jump_mps",
        type=float,
        default=20.0,
        help="|Δego_speed| between consecutive timestamps above this counts as anomaly.",
    )
    ap.add_argument(
        "--max_pos_jump_m",
        type=float,
        default=12.0,
        help="2D ego position step above this counts as anomaly (sorted-by-time consecutive rows).",
    )
    ap.add_argument("--min_rows", type=int, default=80, help="Penalize if row count below this.")
    ap.add_argument("--max_files", type=int, default=0, help="0 = no limit.")
    args = ap.parse_args()

    paths = _discover_csvs(args.data_dir, args.discovery)
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    rows_out = []
    for fp in paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        try:
            with open(fp, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                fn = list(rdr.fieldnames or [])
                rows = list(rdr)
        except Exception as ex:
            rows_out.append(
                {
                    "file": rel,
                    "score_overall": "0",
                    "grade": "F",
                    "issues_summary": "read_error:%s" % type(ex).__name__,
                }
            )
            continue

        m = score_file(
            rows,
            fn,
            args.gap_threshold_sec,
            args.max_speed_jump_mps,
            args.max_pos_jump_m,
            args.min_rows,
        )
        m["file"] = rel
        rows_out.append(m)

    keys = [
        "file",
        "score_overall",
        "grade",
        "n_rows",
        "duration_sec",
        "missing_essential_frac",
        "replay_core_missing_frac",
        "mono_pairs_ratio",
        "duplicate_ts_count",
        "n_dt_gaps_over_thresh",
        "max_dt_sec",
        "mean_dt_sec",
        "n_speed_jumps",
        "n_pos_jumps",
        "ts_none_count",
        "ttc_sentinel_frac",
        "th_sentinel_frac",
        "missing_header_required",
        "rows_missing_any_essential",
        "issues_summary",
    ]

    od = os.path.dirname(os.path.abspath(args.out_csv))
    if od:
        os.makedirs(od, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in keys})

    print("[OK] scored", len(rows_out), "files →", args.out_csv)


if __name__ == "__main__":
    main()
