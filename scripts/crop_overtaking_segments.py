# -*- coding: utf-8 -*-
"""
Crop each compliant overtaking trajectory to:
  [reach_time, t_end]
where reach_time is when lead vehicle reaches target speed (same as extract_overtaking_phases),
and t_end is the last sample where ego_pos_y is minimal (rightmost in-lane) among right-lane
samples after the first return to the right lane, within a window that ends *before* any
second left-lane segment (if the driver overtakes again after 回正, the clip ends at the
first cycle's rightmost following position only).

Output layout mirrors data/:  <out_dir>/T*/行车/<session_folder>/driving_data.csv
Trajectories with left lane before reach_speed are still skipped; multiple post-return
left cycles are accepted and trimmed as above.
"""
from __future__ import print_function

import argparse
import csv
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import extract_overtaking_phases as eop  # noqa: E402


def _participant_from_rel(rel_path):
    """e.g. T9/行车/... -> T9"""
    parts = rel_path.replace("\\", "/").split("/")
    if parts and re.match(r"^T\d+$", parts[0]):
        return parts[0]
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/overtaking_cropped_segments",
    )
    ap.add_argument("--left_y_min", type=float, default=-5.55)
    ap.add_argument("--left_y_max", type=float, default=-2.20)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.95)
    ap.add_argument("--speed_tol_mps", type=float, default=0.3)
    ap.add_argument("--reach_hold_sec", type=float, default=0.7)
    ap.add_argument("--lane_hold_sec", type=float, default=0.5)
    ap.add_argument("--lane_gap_merge_sec", type=float, default=0.3)
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_paths = eop._discover_overtaking_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]

    exp_kmh = {1: 35.0, 2: 50.0, 3: 65.0}
    summary_rows = []
    skipped = []
    written = 0

    for fp in csv_paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        exp_num = eop._parse_exp_num(rel)
        if exp_num is None:
            continue

        target_kmh = exp_kmh[exp_num]
        target_mps = target_kmh / 3.6
        speed_thr = target_mps - args.speed_tol_mps

        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            raw_rows = list(reader)

        if not fieldnames:
            skipped.append((rel, "no_header"))
            continue

        rows_ts = []
        for row in raw_rows:
            ts = eop._parse_float(row, "timestamp")
            ego_y = eop._parse_float(row, "ego_pos_y")
            lead_speed = eop._parse_float(row, "lead_speed")
            if ts is None or ego_y is None or lead_speed is None:
                continue
            frame = row.get("frame", "")
            try:
                frame = int(float(frame))
            except Exception:
                frame = 0
            rows_ts.append((ts, ego_y, lead_speed, frame, row))

        if not rows_ts:
            skipped.append((rel, "no_valid_rows"))
            continue

        rows_ts.sort(key=lambda x: x[0])
        times = [x[0] for x in rows_ts]
        ego_y = [x[1] for x in rows_ts]
        lead_speed = [x[2] for x in rows_ts]
        frames = [x[3] for x in rows_ts]
        aligned_rows = [x[4] for x in rows_ts]

        lane_states = [
            eop.classify_lane(
                y, args.left_y_min, args.left_y_max, args.right_y_min, args.right_y_max
            )
            for y in ego_y
        ]
        n = len(times)

        left_hold_segments = eop._collect_hold_segments(
            times,
            [lane_states[i] == "L" for i in range(n)],
            min_hold_sec=args.lane_hold_sec,
            gap_merge_sec=args.lane_gap_merge_sec,
        )

        lead_flags = [s is not None and s >= speed_thr for s in lead_speed]
        reach_start_idx, _ = eop._gate_segment_by_hold(
            times, lead_flags, min_hold_sec=args.reach_hold_sec, gap_merge_sec=0.3
        )
        if reach_start_idx is None:
            skipped.append((rel, "no_reach_speed"))
            continue

        has_pre_reach_left = False
        for seg_start, seg_end in left_hold_segments:
            if seg_start < reach_start_idx:
                has_pre_reach_left = True
                break
        if has_pre_reach_left:
            skipped.append((rel, "left_before_reach"))
            continue

        left_flags = [
            i >= reach_start_idx and lane_states[i] == "L" for i in range(n)
        ]
        left_start_idx, _ = eop._gate_segment_by_hold(
            times,
            left_flags,
            min_hold_sec=args.lane_hold_sec,
            gap_merge_sec=args.lane_gap_merge_sec,
        )
        if left_start_idx is None:
            skipped.append((rel, "no_left_hold"))
            continue

        right_flags = [
            i >= left_start_idx and lane_states[i] == "R" for i in range(n)
        ]
        right_start_idx, _ = eop._gate_segment_by_hold(
            times,
            right_flags,
            min_hold_sec=args.lane_hold_sec,
            gap_merge_sec=args.lane_gap_merge_sec,
        )
        if right_start_idx is None:
            skipped.append((rel, "no_return_right"))
            continue

        second_left_start_idx = eop._first_left_segment_start_after_idx(
            left_hold_segments, right_start_idx
        )
        window_end_idx = n - 1
        has_second = 0
        if second_left_start_idx is not None:
            window_end_idx = second_left_start_idx - 1
            has_second = 1

        end_idx = eop._fragment_end_idx_rightmost(
            lane_states, ego_y, right_start_idx, window_end_idx
        )
        if end_idx < reach_start_idx:
            skipped.append((rel, "bad_end_idx"))
            continue

        cropped = aligned_rows[reach_start_idx : end_idx + 1]
        if not cropped:
            skipped.append((rel, "empty_crop"))
            continue

        session_dir = os.path.dirname(rel)
        out_sub = os.path.join(args.out_dir, session_dir)
        os.makedirs(out_sub, exist_ok=True)
        out_fp = os.path.join(out_sub, "driving_data.csv")

        with open(out_fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in cropped:
                w.writerow(row)

        written += 1
        summary_rows.append(
            {
                "participant": _participant_from_rel(rel),
                "source_file": rel,
                "out_file": os.path.relpath(out_fp, args.out_dir).replace("\\", "/"),
                "exp": str(exp_num),
                "reach_time": "{:.6f}".format(times[reach_start_idx]),
                "reach_frame": str(frames[reach_start_idx]),
                "end_time": "{:.6f}".format(times[end_idx]),
                "end_frame": str(frames[end_idx]),
                "end_ego_pos_y": "{:.6f}".format(ego_y[end_idx]),
                "has_second_left_after_return": str(has_second),
                "n_rows": str(len(cropped)),
            }
        )

    sum_csv = os.path.join(args.out_dir, "crop_summary.csv")
    if summary_rows:
        keys = list(summary_rows[0].keys())
    else:
        keys = [
            "participant",
            "source_file",
            "out_file",
            "exp",
            "reach_time",
            "reach_frame",
            "end_time",
            "end_frame",
            "end_ego_pos_y",
            "has_second_left_after_return",
            "n_rows",
        ]
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    skip_csv = os.path.join(args.out_dir, "crop_skipped.csv")
    with open(skip_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "reason"])
        w.writeheader()
        for rel, reason in skipped:
            w.writerow({"file": rel, "reason": reason})

    print("[OK] written:", written, "skipped:", len(skipped))
    print("[OK] crop_summary.csv:", sum_csv)
    print("[OK] crop_skipped.csv:", skip_csv)


if __name__ == "__main__":
    main()
