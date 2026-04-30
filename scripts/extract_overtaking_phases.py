# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import os
import re
from collections import defaultdict


def _is_pre_familiarization(path):
    return "pre_familiarization" in path


def _parse_exp_num(path):
    # Match .../exp1_o/... or ...exp2_o...
    m = re.search(r"exp([123])_o", path)
    return int(m.group(1)) if m else None


def _discover_overtaking_csvs(data_dir):
    """
    Discover overtaking trajectories:
      - must match exp[123]_o
      - exclude pre_familiarization
      - exclude *_b
    """
    cands = []
    for root, _, files in os.walk(data_dir):
        if "driving_data.csv" not in files:
            continue
        p = root.replace("\\", "/")
        if _is_pre_familiarization(p):
            continue
        if "/exp" not in p and "exp" not in p:
            continue
        if "_b" in p:
            continue
        if not re.search(r"exp[123]_o", p):
            continue
        # Ensure it's actually overtaking experiment, not other exp_*.
        cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def _parse_float(row, key):
    v = row.get(key, "")
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _gate_segment_by_hold(times, flags, min_hold_sec, gap_merge_sec):
    """
    Given a boolean flag time series, find the first segment start where flags stay True
    for at least min_hold_sec, allowing brief False gaps up to gap_merge_sec.
    Returns index (start_idx) and end_idx (inclusive) or (None, None).
    """
    if not times:
        return None, None
    n = len(times)
    i = 0
    while i < n:
        # Seek a True start
        if not flags[i]:
            i += 1
            continue
        start = i
        true_start_time = times[i]

        # Now accumulate span allowing small gaps
        cur = i
        last_true_idx = i
        last_true_time = times[i]
        # false gap accumulator
        false_gap_start = None
        while cur + 1 < n:
            nxt = cur + 1
            if flags[nxt]:
                last_true_idx = nxt
                last_true_time = times[nxt]
                false_gap_start = None
                cur = nxt
                # check hold by time since true_start_time
                if last_true_time - true_start_time >= min_hold_sec:
                    return start, last_true_idx
            else:
                # outside: start/extend false gap
                if false_gap_start is None:
                    false_gap_start = nxt
                else:
                    # if false gap exceeds merge threshold, stop this candidate segment
                    if times[nxt] - times[false_gap_start] > gap_merge_sec:
                        break
                cur = nxt
        # if we exit, not enough hold
        i = start + 1
    return None, None


def _collect_hold_segments(times, flags, min_hold_sec, gap_merge_sec):
    """
    Collect all hold-valid True segments.
    Returns list of tuples: (start_idx, end_idx) where hold duration >= min_hold_sec.
    """
    segs = []
    if not times:
        return segs

    n = len(times)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue

        start = i
        true_start_time = times[i]
        cur = i
        last_true_idx = i
        false_gap_start = None

        while cur + 1 < n:
            nxt = cur + 1
            if flags[nxt]:
                last_true_idx = nxt
                false_gap_start = None
                cur = nxt
            else:
                if false_gap_start is None:
                    false_gap_start = nxt
                elif times[nxt] - times[false_gap_start] > gap_merge_sec:
                    break
                cur = nxt

        if times[last_true_idx] - true_start_time >= min_hold_sec:
            segs.append((start, last_true_idx))
        i = max(cur + 1, start + 1)

    return segs


def _first_left_segment_start_after_idx(left_hold_segments, after_idx):
    """First left-hold segment start strictly after after_idx, or None."""
    best = None
    for seg_start, seg_end in left_hold_segments:
        if seg_start > after_idx:
            if best is None or seg_start < best:
                best = seg_start
    return best


def _fragment_end_idx_rightmost(
    lane_states, ego_y, right_start_idx, window_end_idx, y_tol=1e-3
):
    """
    Last index in [right_start_idx, window_end_idx] with lane R and ego_y at
    the minimum among R samples in that window (rightmost in-lane).
    """
    n = len(lane_states)
    window_end_idx = min(max(window_end_idx, 0), n - 1)
    if window_end_idx < right_start_idx:
        return right_start_idx
    candidates = [
        i
        for i in range(right_start_idx, window_end_idx + 1)
        if lane_states[i] == "R"
    ]
    if not candidates:
        return right_start_idx
    y_min = min(ego_y[i] for i in candidates)
    last_at_min = None
    for i in candidates:
        if abs(ego_y[i] - y_min) <= y_tol:
            last_at_min = i
    return last_at_min if last_at_min is not None else candidates[-1]


def classify_lane(y, left_y_min, left_y_max, right_y_min, right_y_max):
    if (y >= left_y_min) and (y <= left_y_max):
        return "L"
    if (y >= right_y_min) and (y <= right_y_max):
        return "R"
    return "O"


def summarize_interval(times, ys, lane_states, wanted_state):
    # ratio of wanted_state samples in the interval; assumes times aligned
    if not times:
        return 0.0, 0.0, 0.0
    n = len(times)
    good = 0
    y_min = 1e18
    y_max = -1e18
    for i in range(n):
        if lane_states[i] == wanted_state:
            good += 1
        if ys[i] < y_min:
            y_min = ys[i]
        if ys[i] > y_max:
            y_max = ys[i]
    return float(good) / float(n), y_min, y_max


def make_svg_overtake(
    html_path,
    title,
    series,
    left_bounds,
    right_bounds,
    reach_time,
    left_time,
    right_time,
):
    """
    series: list of (t, ego_y, lane_state, lead_speed)
    left_bounds/right_bounds: (min,max)
    """
    if not series:
        return

    t_vals = [p[0] for p in series]
    y_vals = [p[1] for p in series]
    lead_vals = [p[3] for p in series if p[3] is not None]
    t_min = min(t_vals)
    t_max = max(t_vals)
    y_min = min(y_vals)
    y_max = max(y_vals)

    # padding
    pad_t = 0.02 * (t_max - t_min) if t_max > t_min else 1.0
    pad_y = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    t_min -= pad_t
    t_max += pad_t
    y_min -= pad_y
    y_max += pad_y

    W, H = 1100, 520
    left_margin = 80
    top_margin = 30
    plot_w = W - left_margin - 20
    plot_h = H - top_margin - 20

    def x_of(t):
        if t_max == t_min:
            return left_margin + plot_w / 2.0
        return left_margin + (t - t_min) / (t_max - t_min) * plot_w

    def y_of(y):
        if y_max == y_min:
            return top_margin + plot_h / 2.0
        return top_margin + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    # boundaries
    left_ymin, left_ymax = left_bounds
    right_ymin, right_ymax = right_bounds

    left_ymin_line = y_of(left_ymin)
    left_ymax_line = y_of(left_ymax)
    right_ymin_line = y_of(right_ymin)
    right_ymax_line = y_of(right_ymax)

    # Pre-compute SVG label coordinates.
    # Note: Python str.format does not support arithmetic expressions inside { }.
    lm_plus_5 = left_margin + 5
    top_plus_15 = top_margin + 15
    top_plus_ph_plus_18 = top_margin + plot_h + 18

    # points
    grey = []
    red = []
    green = []
    for (t, ego_y, lane_state, lead_speed) in series:
        pt = (x_of(t), y_of(ego_y))
        if lane_state == "R":
            red.append(pt)     # right lane points
        elif lane_state == "L":
            green.append(pt)   # left lane points
        else:
            grey.append(pt)

    def pts_to_str(pts):
        return " ".join(["{:.2f},{:.2f}".format(p[0], p[1]) for p in pts])

    def line_at(time, color, width):
        if time is None:
            return ""
        x = x_of(time)
        return '<line x1="{:.2f}" y1="{}" x2="{:.2f}" y2="{}" stroke="{}" stroke-width="{}" opacity="0.9"/>'.format(
            x, top_margin, x, top_margin + plot_h, color, width
        )

    html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
</head>
<body style="font-family:sans-serif;">
<h3>{title}</h3>
<div style="color:#444;margin-bottom:8px;">
  left_y=[{lymin:.2f},{lymax:.2f}] right_y=[{rymin:.2f},{rymax:.2f}]<br/>
  reach_time={reach_time:.3f} left_hold={left_time:.3f} right_hold={right_time:.3f}
</div>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="border:1px solid #ddd;background:#fff;">
  <line x1="{lm}" y1="{lyminl:.2f}" x2="{lm2:.2f}" y2="{lyminl:.2f}" stroke="#1f77b4" stroke-width="2" opacity="0.7"/>
  <line x1="{lm}" y1="{lymaxl:.2f}" x2="{lm2:.2f}" y2="{lymaxl:.2f}" stroke="#1f77b4" stroke-width="2" opacity="0.7"/>
  <line x1="{lm}" y1="{ryminl:.2f}" x2="{lm2:.2f}" y2="{ryminl:.2f}" stroke="#ff7f0e" stroke-width="2" opacity="0.7"/>
  <line x1="{lm}" y1="{rymaxl:.2f}" x2="{lm2:.2f}" y2="{rymaxl:.2f}" stroke="#ff7f0e" stroke-width="2" opacity="0.7"/>

  {reach_line}
  {left_line}
  {right_line}

  <polyline fill="none" stroke="#999" stroke-width="1.2" points="{grey}" opacity="0.45"/>
  <polyline fill="none" stroke="#d62728" stroke-width="2.0" points="{red}" opacity="0.70"/>
  <polyline fill="none" stroke="#2ca02c" stroke-width="2.0" points="{green}" opacity="0.70"/>

  <text x="{lm_plus_5}" y="{top_plus_15}" fill="#333" font-size="12">timestamp</text>
  <text x="10" y="{top_plus_ph_plus_18}" fill="#333" font-size="12">ego_pos_y (m)</text>
</svg>
</body></html>
""".format(
        title=title,
        lymin=left_ymin,
        lymax=left_ymax,
        rymin=right_ymin,
        rymax=right_ymax,
        reach_time=(reach_time if reach_time is not None else -1.0),
        left_time=(left_time if left_time is not None else -1.0),
        right_time=(right_time if right_time is not None else -1.0),
        W=W,
        H=H,
        lm=left_margin,
        lm2=left_margin + plot_w,
        lyminl=left_ymin_line,
        lymaxl=left_ymax_line,
        ryminl=right_ymin_line,
        rymaxl=right_ymax_line,
        grey=pts_to_str(grey),
        red=pts_to_str(red),
        green=pts_to_str(green),
        reach_line=line_at(reach_time, "#9467bd", 3),
        left_line=line_at(left_time, "#1f77b4", 3),
        right_line=line_at(right_time, "#ff7f0e", 3),
        top=top_margin,
        ph=plot_h,
        lm_plus_5=lm_plus_5,
        top_plus_15=top_plus_15,
        top_plus_ph_plus_18=top_plus_ph_plus_18,
    )

    out_dir = os.path.dirname(html_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    with open(html_path, "w") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument("--out_dir", type=str, default="/home/zwx/driver_model/outputs/overtaking_phases")

    # Lane bounds (ego_pos_y)
    ap.add_argument("--left_y_min", type=float, default=-5.55)
    ap.add_argument("--left_y_max", type=float, default=-2.20)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.95)

    # Lead speed threshold
    ap.add_argument("--speed_tol_mps", type=float, default=0.3)
    ap.add_argument("--reach_hold_sec", type=float, default=0.7)

    # Lane hold detection
    ap.add_argument("--lane_hold_sec", type=float, default=0.5)
    ap.add_argument("--lane_gap_merge_sec", type=float, default=0.3)

    ap.add_argument("--save_html", action="store_true", default=True)
    ap.add_argument("--no_save_html", action="store_true", default=False)
    ap.add_argument("--max_files", type=int, default=0, help="0 means all")

    args = ap.parse_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    html_dir = os.path.join(args.out_dir, "html")
    if args.save_html and (not args.no_save_html):
        if not os.path.isdir(html_dir):
            os.makedirs(html_dir)
    non_compliant_dir = os.path.join(args.out_dir, "non_compliant")
    non_compliant_html_dir = os.path.join(non_compliant_dir, "html")
    if not os.path.isdir(non_compliant_dir):
        os.makedirs(non_compliant_dir)
    if args.save_html and (not args.no_save_html):
        if not os.path.isdir(non_compliant_html_dir):
            os.makedirs(non_compliant_html_dir)

    csv_paths = _discover_overtaking_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]

    # exp targets in km/h
    exp_kmh = {1: 35.0, 2: 50.0, 3: 65.0}

    summary_rows = []
    rejected_rows = []
    files_processed = 0
    files_with_phases = 0
    rejected_by_constraints = 0

    for fp in csv_paths:
        files_processed += 1
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        exp_num = _parse_exp_num(rel)
        if exp_num is None:
            continue
        target_kmh = exp_kmh[exp_num]
        target_mps = target_kmh / 3.6
        speed_thr = target_mps - args.speed_tol_mps

        # Load
        rows = []
        with open(fp, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                ts = _parse_float(row, "timestamp")
                ego_y = _parse_float(row, "ego_pos_y")
                lead_speed = _parse_float(row, "lead_speed")
                frame = row.get("frame", "")
                if frame is None:
                    frame = ""
                try:
                    frame = int(float(frame))
                except Exception:
                    frame = 0
                if ts is None or ego_y is None or lead_speed is None:
                    continue
                rows.append((ts, ego_y, lead_speed, frame))

        if not rows:
            continue

        # Ensure ordered by time
        rows.sort(key=lambda x: x[0])
        times = [x[0] for x in rows]
        ego_y = [x[1] for x in rows]
        lead_speed = [x[2] for x in rows]
        frames = [x[3] for x in rows]

        lane_states = [classify_lane(y, args.left_y_min, args.left_y_max, args.right_y_min, args.right_y_max) for y in ego_y]

        left_hold_segments = _collect_hold_segments(
            times,
            [lane_states[i] == "L" for i in range(len(rows))],
            min_hold_sec=args.lane_hold_sec,
            gap_merge_sec=args.lane_gap_merge_sec,
        )

        # Find reach_time: lead_speed >= speed_thr for hold period
        lead_flags = [s is not None and s >= speed_thr for s in lead_speed]
        reach_start_idx, reach_end_idx = _gate_segment_by_hold(
            times, lead_flags, min_hold_sec=args.reach_hold_sec, gap_merge_sec=0.3
        )
        if reach_start_idx is None:
            continue
        reach_time = times[reach_start_idx]
        reach_frame = frames[reach_start_idx]
        series = []
        for i in range(len(rows)):
            series.append((times[i], ego_y[i], lane_states[i], lead_speed[i]))

        safe = re.sub(r"[^0-9a-zA-Z._-]+", "_", rel)

        # Constraint 1: No effective left-lane hold segment is allowed before reach_time.
        has_pre_reach_left = False
        pre_reach_left_idx = None
        for seg_start, seg_end in left_hold_segments:
            if seg_start < reach_start_idx:
                has_pre_reach_left = True
                pre_reach_left_idx = seg_start
                break
        if has_pre_reach_left:
            rejected_by_constraints += 1
            rejected_rows.append({
                "file": rel,
                "exp": str(exp_num),
                "reason": "left_lane_change_before_reach_speed",
                "reach_time": "{:.6f}".format(reach_time),
                "reach_frame": str(reach_frame),
                "detail_time": "{:.6f}".format(times[pre_reach_left_idx]) if pre_reach_left_idx is not None else "",
                "detail_frame": str(frames[pre_reach_left_idx]) if pre_reach_left_idx is not None else "",
            })
            if args.save_html and (not args.no_save_html):
                html_path = os.path.join(non_compliant_html_dir, safe + ".html")
                make_svg_overtake(
                    html_path=html_path,
                    title=rel + " [REJECTED: before_reach]",
                    series=series,
                    left_bounds=(args.left_y_min, args.left_y_max),
                    right_bounds=(args.right_y_min, args.right_y_max),
                    reach_time=reach_time,
                    left_time=(times[pre_reach_left_idx] if pre_reach_left_idx is not None else None),
                    right_time=None,
                )
            continue

        # After reach_time, find left_hold
        left_flags = [i >= reach_start_idx and lane_states[i] == "L" for i in range(len(rows))]
        left_start_idx, left_end_idx = _gate_segment_by_hold(
            times, left_flags, min_hold_sec=args.lane_hold_sec, gap_merge_sec=args.lane_gap_merge_sec
        )
        if left_start_idx is None:
            continue
        left_time = times[left_start_idx]
        left_frame = frames[left_start_idx]

        # After left_hold, find right_hold
        right_flags = [i >= left_start_idx and lane_states[i] == "R" for i in range(len(rows))]
        right_start_idx, right_end_idx = _gate_segment_by_hold(
            times, right_flags, min_hold_sec=args.lane_hold_sec, gap_merge_sec=args.lane_gap_merge_sec
        )
        if right_start_idx is None:
            continue
        right_time = times[right_start_idx]
        right_frame = frames[right_start_idx]

        # After first return-to-right, if driver goes left again (second overtake),
        # trim the valid fragment to end before that second left segment: still a valid first-cycle clip.
        second_left_start_idx = _first_left_segment_start_after_idx(
            left_hold_segments, right_start_idx
        )
        window_end_idx = len(rows) - 1
        has_second_cycle = 0
        if second_left_start_idx is not None:
            window_end_idx = second_left_start_idx - 1
            has_second_cycle = 1
        fragment_end_idx = _fragment_end_idx_rightmost(
            lane_states, ego_y, right_start_idx, window_end_idx
        )
        fragment_end_time = times[fragment_end_idx]
        fragment_end_frame = frames[fragment_end_idx]

        files_with_phases += 1

        follow_start_idx = 0
        # Use first right lane sample before reach as follow start (best-effort)
        for i in range(0, reach_start_idx + 1):
            if lane_states[i] == "R":
                follow_start_idx = i
                break
        follow_end_idx = reach_start_idx

        # Compute ratios inside lane for each phase
        def slice_ratio(a, b, wanted):
            if b < a:
                return 0.0, 0.0, 0.0
            sub_lane = lane_states[a:b + 1]
            sub_y = ego_y[a:b + 1]
            n = len(sub_lane)
            good = 0
            y_min = 1e18
            y_max = -1e18
            for j in range(n):
                if sub_lane[j] == wanted:
                    good += 1
                y_min = min(y_min, sub_y[j])
                y_max = max(y_max, sub_y[j])
            return float(good) / float(n), y_min, y_max

        follow_ratio_R, follow_ymin, follow_ymax = slice_ratio(follow_start_idx, follow_end_idx, "R")
        left_ratio_L, left_ymin, left_ymax = slice_ratio(left_start_idx, right_start_idx, "L")
        ret_ratio_R, ret_ymin, ret_ymax = slice_ratio(
            right_start_idx, fragment_end_idx, "R"
        )

        summary_rows.append({
            "file": rel,
            "exp": str(exp_num),
            "target_kmh": "{:.3f}".format(target_kmh),
            "target_mps": "{:.6f}".format(target_mps),
            "reach_time": "{:.6f}".format(reach_time),
            "reach_frame": str(reach_frame),
            "left_time": "{:.6f}".format(left_time),
            "left_frame": str(left_frame),
            "right_time": "{:.6f}".format(right_time),
            "right_frame": str(right_frame),
            "fragment_end_time": "{:.6f}".format(fragment_end_time),
            "fragment_end_frame": str(fragment_end_frame),
            "has_second_left_after_return": str(has_second_cycle),
            "follow_right_dur": "{:.6f}".format(times[follow_end_idx] - times[follow_start_idx]),
            "to_left_dur": "{:.6f}".format(left_time - reach_time),
            "left_overtake_dur": "{:.6f}".format(right_time - left_time),
            "return_right_dur": "{:.6f}".format(fragment_end_time - right_time),
            "follow_right_ratio": "{:.6f}".format(follow_ratio_R),
            "left_lane_ratio": "{:.6f}".format(left_ratio_L),
            "return_right_ratio": "{:.6f}".format(ret_ratio_R),
            "follow_ymin": "{:.6f}".format(follow_ymin),
            "follow_ymax": "{:.6f}".format(follow_ymax),
            "left_ymin": "{:.6f}".format(left_ymin),
            "left_ymax": "{:.6f}".format(left_ymax),
            "return_ymin": "{:.6f}".format(ret_ymin),
            "return_ymax": "{:.6f}".format(ret_ymax),
        })

        if args.save_html and (not args.no_save_html):
            # Make a safe html filename
            html_path = os.path.join(html_dir, safe + ".html")
            make_svg_overtake(
                html_path=html_path,
                title=rel,
                series=series,
                left_bounds=(args.left_y_min, args.left_y_max),
                right_bounds=(args.right_y_min, args.right_y_max),
                reach_time=reach_time,
                left_time=left_time,
                right_time=right_time,
            )

    out_csv = os.path.join(args.out_dir, "segments_summary.csv")
    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "file", "exp", "target_kmh", "target_mps",
        "reach_time", "reach_frame", "left_time", "left_frame", "right_time", "right_frame",
        "fragment_end_time", "fragment_end_frame", "has_second_left_after_return",
        "follow_right_dur", "to_left_dur", "left_overtake_dur", "return_right_dur",
        "follow_right_ratio", "left_lane_ratio", "return_right_ratio",
        "follow_ymin", "follow_ymax", "left_ymin", "left_ymax", "return_ymin", "return_ymax"
    ]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    rejected_csv = os.path.join(non_compliant_dir, "non_compliant_summary.csv")
    rejected_fields = ["file", "exp", "reason", "reach_time", "reach_frame", "detail_time", "detail_frame"]
    with open(rejected_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=rejected_fields)
        w.writeheader()
        for row in rejected_rows:
            w.writerow(row)

    print("[OK] processed:", files_processed, "with phases:", files_with_phases, "rejected_by_constraints:", rejected_by_constraints)
    print("[OK] segments_summary.csv:", out_csv)
    print("[OK] non_compliant_summary.csv:", rejected_csv)


if __name__ == "__main__":
    main()

