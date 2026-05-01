# -*- coding: utf-8 -*-
"""
Clean following (跟驰) trajectories for imitation learning.

Design principles tailored to behaviour cloning of car-following:
- Treat (throttle, brake, steer, longitudinal_control, ego_speed, ego_acceleration,
  ego_jerk, ego_pos_*, ego_yaw) as the EXPERT'S RECORDED ACTIONS / STATE: never
  smooth or interpolate them. We may DROP rows or SPLIT segments, never edit values.
- Treat all lead_* / distance_headway / time_headway / relative_speed / ttc as the
  recorded leading-vehicle situation that we must reproduce exactly during replay,
  so we keep them untouched (only mask sentinel 999).
- Keep the data as a list of contiguous SEGMENTS so the model is only trained on
  truly continuous time series; gaps from simulator stalls do not get glued
  together silently.

Pipeline:
  1. Drop rows missing any essential field.
  2. Sort by timestamp; drop duplicates / non-monotonic rows.
  3. Mask sentinel 999 in ttc / time_headway -> empty + boolean *_valid column.
  4. Split into contiguous segments where dt <= gap_threshold_sec.
  5. Trim leading and trailing "stationary / no-interaction" rows
     (both ego and lead under v_min for too long).
  6. Drop segments shorter than min_segment_duration_sec.
  7. Drop rows whose physical signals are impossible (huge instantaneous jumps);
     segment is split at those locations.
  8. Drop segments where the lead is essentially absent throughout
     (distance_headway always above max_useful_headway and lead_speed ~ 0).

Output layout (mirrors data/):
  <out_dir>/T*/行车/<session>/segment_001.csv, segment_002.csv, ...
  <out_dir>/cleaning_summary.csv          - one row per kept segment
  <out_dir>/cleaning_dropped.csv          - one row per dropped source file
  
  python3 /home/zwx/driver_model/following/scripts/clean_following_for_imitation.py \
  --data_dir /home/zwx/driver_model/following/outputs/following_calibrated/T9 \
  --out_dir /home/zwx/driver_model/following/outputs/following_il_clean_gap04/T9
  
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import re

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


def discover_following_csvs(data_dir):
    cands = []
    for root, _, files in os.walk(data_dir):
        if "driving_data.csv" not in files:
            continue
        p = root.replace("\\", "/")
        if "pre_familiarization" in p:
            continue
        if "overtaking" in p:
            continue
        if re.search(r"/[^/]*_o(?:/|$)", p):
            continue
        if "_b" in p:
            continue
        if ("following" in p) or re.search(r"/[^/]*_f(?:/|$)", p):
            cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def _normalize_sentinels(rows):
    """ttc / time_headway with value 999 are sentinels for 'no lead in front';
    convert them to empty and add *_valid columns (Yes/No)."""
    for r in rows:
        for k in ("ttc", "time_headway"):
            v = _parse_float(r.get(k))
            if v is not None and abs(v - 999.0) < 1e-6:
                r[k] = ""
                r[k + "_valid"] = "0"
            else:
                r[k + "_valid"] = "1" if v is not None else "0"
    return rows


def _drop_missing_essentials(rows):
    out = []
    for r in rows:
        ok = True
        for k in ESSENTIAL_FIELDS:
            if _parse_float(r.get(k)) is None:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _sort_dedup_monotonic(rows):
    rows.sort(key=lambda r: _parse_float(r.get("timestamp")) or 0.0)
    out = []
    last_ts = None
    for r in rows:
        ts = _parse_float(r.get("timestamp"))
        if ts is None:
            continue
        if last_ts is not None and ts <= last_ts:
            continue
        out.append(r)
        last_ts = ts
    return out


def _split_by_time_gap(rows, gap_threshold_sec):
    if not rows:
        return []
    segs = []
    cur = [rows[0]]
    last_ts = _parse_float(rows[0].get("timestamp"))
    for r in rows[1:]:
        ts = _parse_float(r.get("timestamp"))
        if ts is None or last_ts is None:
            cur.append(r)
            last_ts = ts
            continue
        if ts - last_ts > gap_threshold_sec:
            segs.append(cur)
            cur = [r]
        else:
            cur.append(r)
        last_ts = ts
    if cur:
        segs.append(cur)
    return segs


def _trim_stationary_edges(rows, v_min, max_grace_sec):
    """Trim leading/trailing rows where both ego_speed and lead_speed < v_min
    for longer than max_grace_sec."""
    if not rows:
        return rows

    def _is_stationary(r):
        ev = _parse_float(r.get("ego_speed")) or 0.0
        lv = _parse_float(r.get("lead_speed")) or 0.0
        return (ev < v_min) and (lv < v_min)

    n = len(rows)
    start = 0
    while start < n and _is_stationary(rows[start]):
        start += 1
    if start >= n:
        return []
    if start > 0:
        ts0 = _parse_float(rows[0].get("timestamp")) or 0.0
        ts_active = _parse_float(rows[start].get("timestamp")) or ts0
        if ts_active - ts0 < max_grace_sec:
            start = 0

    end = n - 1
    while end >= start and _is_stationary(rows[end]):
        end -= 1
    if end < start:
        return []
    if end < n - 1:
        ts_last = _parse_float(rows[n - 1].get("timestamp")) or 0.0
        ts_inactive = _parse_float(rows[end].get("timestamp")) or ts_last
        if ts_last - ts_inactive < max_grace_sec:
            end = n - 1

    return rows[start : end + 1]


def _split_on_physical_jumps(rows, max_speed_jump, max_pos_jump):
    """Split segment at rows that have implausible jumps in ego_speed or ego_pos_x.
    The offending row is dropped (treated as a sensor glitch), the segment splits there."""
    if len(rows) < 2:
        return [rows]
    segs = []
    cur = [rows[0]]
    for i in range(1, len(rows)):
        prev = rows[i - 1]
        cur_row = rows[i]
        bad = False
        ev_prev = _parse_float(prev.get("ego_speed"))
        ev_cur = _parse_float(cur_row.get("ego_speed"))
        if (
            ev_prev is not None
            and ev_cur is not None
            and abs(ev_cur - ev_prev) > max_speed_jump
        ):
            bad = True
        ex_prev = _parse_float(prev.get("ego_pos_x"))
        ex_cur = _parse_float(cur_row.get("ego_pos_x"))
        ey_prev = _parse_float(prev.get("ego_pos_y"))
        ey_cur = _parse_float(cur_row.get("ego_pos_y"))
        if (
            ex_prev is not None
            and ex_cur is not None
            and ey_prev is not None
            and ey_cur is not None
        ):
            jump = math.hypot(ex_cur - ex_prev, ey_cur - ey_prev)
            if jump > max_pos_jump:
                bad = True
        if bad:
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(cur_row)
    if cur:
        segs.append(cur)
    return [s for s in segs if s]


def _has_meaningful_lead(rows, max_useful_headway, v_min):
    """Drop a segment that has effectively no lead interaction:
    distance_headway always >= max_useful_headway AND lead_speed < v_min for the entire segment."""
    if not rows:
        return False
    saw_close = False
    saw_lead_moving = False
    for r in rows:
        d = _parse_float(r.get("distance_headway"))
        lv = _parse_float(r.get("lead_speed")) or 0.0
        if d is not None and d < max_useful_headway:
            saw_close = True
        if lv >= v_min:
            saw_lead_moving = True
        if saw_close or saw_lead_moving:
            return True
    return False


def _segment_duration(rows):
    if not rows:
        return 0.0
    t0 = _parse_float(rows[0].get("timestamp")) or 0.0
    t1 = _parse_float(rows[-1].get("timestamp")) or 0.0
    return max(0.0, t1 - t0)


def clean_session(rows, args):
    """Run pipeline on one source file's rows.

    Returns:
      (segments, diag) where
        - segments: kept segments (list[list[dict]])
        - diag: per-session diagnostic counters
    """
    rows = _drop_missing_essentials(rows)
    rows = _sort_dedup_monotonic(rows)
    rows = _normalize_sentinels(rows)

    base_segs = _split_by_time_gap(rows, args.gap_threshold_sec)
    diag = {
        "time_gap_segments": len(base_segs),
        "jump_splits": 0,
        "too_short": 0,
        "no_lead": 0,
        "kept": 0,
    }

    final = []
    for seg in base_segs:
        seg = _trim_stationary_edges(seg, args.v_min_mps, args.startup_grace_sec)
        if not seg:
            continue
        sub_segs = _split_on_physical_jumps(
            seg, args.max_speed_jump_mps, args.max_pos_jump_m
        )
        if len(sub_segs) > 1:
            diag["jump_splits"] += len(sub_segs) - 1
        for sub in sub_segs:
            if _segment_duration(sub) < args.min_segment_duration_sec:
                diag["too_short"] += 1
                continue
            if not _has_meaningful_lead(sub, args.max_useful_headway_m, args.v_min_mps):
                diag["no_lead"] += 1
                continue
            final.append(sub)
            diag["kept"] += 1
    return final, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/outputs/following_calibrated")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/following_il_clean",
    )
    ap.add_argument("--gap_threshold_sec", type=float, default=0.4)
    ap.add_argument("--min_segment_duration_sec", type=float, default=5.0)
    ap.add_argument("--startup_grace_sec", type=float, default=2.0)
    ap.add_argument("--v_min_mps", type=float, default=0.5)
    ap.add_argument("--max_speed_jump_mps", type=float, default=6.0,
                    help="Sample-to-sample |Δego_speed| above this is treated as a glitch")
    ap.add_argument("--max_pos_jump_m", type=float, default=4.0,
                    help="Sample-to-sample ego position jump above this is treated as a glitch")
    ap.add_argument("--max_useful_headway_m", type=float, default=200.0)
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = discover_following_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    summary = []
    dropped = []
    diagnostics = []

    for fp in paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if not fieldnames:
            dropped.append({"file": rel, "reason": "no_header"})
            continue

        added_fields = []
        for k in ("ttc_valid", "time_headway_valid"):
            if k not in fieldnames:
                added_fields.append(k)
        out_fields = fieldnames + added_fields

        segments, diag = clean_session(rows, args)
        diagnostics.append({
            "source_file": rel,
            "rows_in": str(len(rows)),
            "time_gap_segments": str(diag["time_gap_segments"]),
            "jump_splits": str(diag["jump_splits"]),
            "too_short": str(diag["too_short"]),
            "no_lead": str(diag["no_lead"]),
            "kept": str(diag["kept"]),
        })

        if not segments:
            dropped.append({"file": rel, "reason": "no_valid_segment"})
            continue

        session_dir = os.path.dirname(rel)
        out_session_dir = os.path.join(args.out_dir, session_dir)
        if not os.path.isdir(out_session_dir):
            os.makedirs(out_session_dir)

        for i, seg in enumerate(segments, start=1):
            out_fp = os.path.join(out_session_dir, "segment_{:03d}.csv".format(i))
            with open(out_fp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=out_fields)
                w.writeheader()
                for r in seg:
                    for k in added_fields:
                        if k not in r:
                            r[k] = ""
                    w.writerow(r)

            t0 = _parse_float(seg[0].get("timestamp")) or 0.0
            t1 = _parse_float(seg[-1].get("timestamp")) or 0.0
            summary.append({
                "source_file": rel,
                "out_file": os.path.relpath(out_fp, args.out_dir).replace("\\", "/"),
                "n_rows": str(len(seg)),
                "duration_sec": "{:.6f}".format(t1 - t0),
                "start_ts": "{:.6f}".format(t0),
                "end_ts": "{:.6f}".format(t1),
            })

    sum_csv = os.path.join(args.out_dir, "cleaning_summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source_file", "out_file", "n_rows", "duration_sec", "start_ts", "end_ts"],
        )
        w.writeheader()
        for row in summary:
            w.writerow(row)

    drop_csv = os.path.join(args.out_dir, "cleaning_dropped.csv")
    with open(drop_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "reason"])
        w.writeheader()
        for row in dropped:
            w.writerow(row)

    diag_csv = os.path.join(args.out_dir, "cleaning_diagnostics.csv")
    with open(diag_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "rows_in",
                "time_gap_segments",
                "jump_splits",
                "too_short",
                "no_lead",
                "kept",
            ],
        )
        w.writeheader()
        for row in diagnostics:
            w.writerow(row)

    print("[OK] kept segments:", len(summary), " dropped files:", len(dropped))
    print("[OK] cleaning_summary.csv:", sum_csv)
    print("[OK] cleaning_dropped.csv:", drop_csv)
    print("[OK] cleaning_diagnostics.csv:", diag_csv)


if __name__ == "__main__":
    main()
