import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

'''

'''

@dataclass
class Segment:
    start_ts: float
    end_ts: float
    start_frame: int
    end_frame: int
    duration_sec: float
    min_y: float
    max_y: float
    violation_ratio: float
    violation_type: str  # "too_left" / "too_right" / "out_of_bounds"


def _sanitize_filename(s: str) -> str:
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
    return s.strip("_")


def discover_following_csvs(data_dir: str) -> List[str]:
    """
    Discover candidate following trajectories.
    Includes:
      - folders containing "_f" (e.g. exp1_f)
      - folders containing "following" (e.g. seg001_following)
    Excludes:
      - overtaking folders containing "overtaking"
      - folders containing "_o" (e.g. exp2_o)
    """
    cands: List[str] = []
    for root, _, files in os.walk(data_dir):
        if "driving_data.csv" not in files:
            continue
        # root is the folder containing driving_data.csv
        p = root.replace("\\", "/")
        # user request: ignore pre_familiarization
        if "pre_familiarization" in p:
            continue
        if "overtaking" in p:
            continue
        # exp*_o or any folder suffix "_o"
        if re.search(r"/[^/]*_o(?:/|$)", p):
            continue

        include = ("following" in p) or re.search(r"/[^/]*_f(?:/|$)", p)
        if include:
            cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
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


def parse_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    v = row.get(key, "")
    if v is None:
        return default
    v = v.strip()
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def detect_violation_segments(
    rows: List[Dict[str, str]],
    right_y_min: float,
    right_y_max: float,
    gap_threshold_sec: float,
    min_segment_duration_sec: float,
) -> Tuple[List[Segment], List[Tuple[float, float, bool]]]:
    """
    Returns:
      - violation segments (continuous outside-range intervals)
      - point series for visualization: (timestamp, ego_pos_y, is_outside)
    """
    ts_y_outside: List[Tuple[float, float, bool]] = []
    segs: List[Segment] = []

    # Build time series (skip rows with missing fields)
    series: List[Tuple[float, float, int]] = []
    for r in rows:
        ts = parse_float(r, "timestamp")
        y = parse_float(r, "ego_pos_y")
        if ts is None or y is None:
            continue
        series.append((ts, y, parse_int(r, "frame", default=0)))

    if not series:
        return [], []

    prev_outside = None
    cur_start_idx = None
    cur_min_y = 0.0
    cur_max_y = 0.0
    cur_start_ts = 0.0
    cur_start_frame = 0
    cur_violation_count = 0
    cur_total_count = 0

    def outside_flag(y: float) -> bool:
        return (y < right_y_min) or (y > right_y_max)

    for i, (ts, y, frame) in enumerate(series):
        out = outside_flag(y)
        ts_y_outside.append((ts, y, out))

        if out:
            if cur_start_idx is None:
                # start a new segment
                cur_start_idx = i
                cur_start_ts = ts
                cur_start_frame = frame
                cur_min_y = y
                cur_max_y = y
                cur_violation_count = 1
                cur_total_count = 1
            else:
                # split if there's a time gap (prevents merging across reboots / stalls)
                prev_ts = series[i - 1][0]
                dt = ts - prev_ts
                if dt > gap_threshold_sec:
                    # start a new segment
                    cur_violation_count = 1
                    cur_total_count = 1
                    cur_min_y = y
                    cur_max_y = y
                    cur_start_ts = ts
                    cur_start_frame = frame
                    cur_start_idx = i
                else:
                    cur_violation_count += 1
                    cur_total_count += 1
                    cur_min_y = min(cur_min_y, y)
                    cur_max_y = max(cur_max_y, y)
        else:
            # if we were in a segment, close it
            if cur_start_idx is not None:
                # close with end at previous index (inside row belongs outside? no, so end is i-1)
                end_ts = series[i - 1][0]
                end_frame = series[i - 1][2]
                duration = end_ts - cur_start_ts
                if duration >= min_segment_duration_sec:
                    # Determine violation type by min/max relative to bounds
                    if cur_max_y > right_y_max and cur_min_y < right_y_min:
                        vtype = "out_of_bounds"
                    elif cur_max_y > right_y_max:
                        vtype = "too_right"
                    else:
                        vtype = "too_left"
                    segs.append(
                        Segment(
                            start_ts=cur_start_ts,
                            end_ts=end_ts,
                            start_frame=cur_start_frame,
                            end_frame=end_frame,
                            duration_sec=duration,
                            min_y=cur_min_y,
                            max_y=cur_max_y,
                            # This segment is defined as all points being outside-range,
                            # so the ratio is effectively 1.0 (kept for compatibility).
                            violation_ratio=1.0,
                            violation_type=vtype,
                        )
                    )
                cur_start_idx = None
                prev_outside = False
            # else stay idle

        prev_outside = out

    # close if ends outside
    if cur_start_idx is not None:
        end_ts = series[-1][0]
        end_frame = series[-1][2]
        duration = end_ts - cur_start_ts
        if duration >= min_segment_duration_sec:
            if cur_max_y > right_y_max and cur_min_y < right_y_min:
                vtype = "out_of_bounds"
            elif cur_max_y > right_y_max:
                vtype = "too_right"
            else:
                vtype = "too_left"
            segs.append(
                Segment(
                    start_ts=cur_start_ts,
                    end_ts=end_ts,
                    start_frame=cur_start_frame,
                    end_frame=end_frame,
                    duration_sec=duration,
                    min_y=cur_min_y,
                    max_y=cur_max_y,
                    violation_ratio=1.0,
                    violation_type=vtype,
                )
            )

    return segs, ts_y_outside


def export_segment_csv(
    src_rows: List[Dict[str, str]],
    out_csv_path: str,
    seg_start_ts: float,
    seg_end_ts: float,
):
    if not src_rows:
        return
    fieldnames = list(src_rows[0].keys())
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in src_rows:
            ts = parse_float(r, "timestamp")
            if ts is None:
                continue
            if (ts >= seg_start_ts) and (ts <= seg_end_ts):
                w.writerow(r)


def make_svg_html(
    html_path: str,
    title: str,
    series: List[Tuple[float, float, bool]],
    right_y_min: float,
    right_y_max: float,
):
    if not series:
        return

    ts_vals = [p[0] for p in series]
    y_vals = [p[1] for p in series]
    t_min, t_max = min(ts_vals), max(ts_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Add margin for better view
    pad_t = max(1e-6, 0.02 * (t_max - t_min) if t_max > t_min else 1.0)
    pad_y = max(1e-6, 0.05 * (y_max - y_min) if y_max > y_min else 1.0)
    t_min -= pad_t
    t_max += pad_t
    y_min -= pad_y
    y_max += pad_y

    def x_of(t: float, w: int) -> float:
        if t_max == t_min:
            return w / 2.0
        return (t - t_min) / (t_max - t_min) * w

    def y_of(y: float, h: int) -> float:
        if y_max == y_min:
            return h / 2.0
        # SVG y grows downward
        return (1.0 - (y - y_min) / (y_max - y_min)) * h

    W, H = 1000, 420
    left_margin = 70
    top_margin = 30
    plot_w = W - left_margin - 20
    plot_h = H - top_margin - 20

    def x_plot(t: float) -> float:
        return left_margin + x_of(t, plot_w)

    def y_plot(y: float) -> float:
        return top_margin + y_of(y, plot_h)

    # Build polylines
    grey_pts = []
    red_pts = []
    for ts, y, outside in series:
        pt = (x_plot(ts), y_plot(y))
        if outside:
            red_pts.append(pt)
        else:
            grey_pts.append(pt)

    # Convert points to "x,y x,y ..."
    def pts_to_str(pts: List[Tuple[float, float]]) -> str:
        return " ".join([f"{p[0]:.2f},{p[1]:.2f}" for p in pts])

    y_min_line = y_plot(right_y_min)
    y_max_line = y_plot(right_y_max)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
</head>
<body style="font-family: sans-serif;">
<h3>{title}</h3>
<div style="color:#444;margin-bottom:8px;">
  右车道范围(ego_pos_y): [{right_y_min:.2f}, {right_y_max:.2f}]
</div>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="border:1px solid #ddd;background:#fff;">
  <!-- boundaries -->
  <line x1="{left_margin}" y1="{y_min_line:.2f}" x2="{left_margin+plot_w:.2f}" y2="{y_min_line:.2f}" stroke="#1f77b4" stroke-width="2" opacity="0.9"/>
  <line x1="{left_margin}" y1="{y_max_line:.2f}" x2="{left_margin+plot_w:.2f}" y2="{y_max_line:.2f}" stroke="#1f77b4" stroke-width="2" opacity="0.9"/>
  <text x="{left_margin+5}" y="{y_min_line-6:.2f}" fill="#1f77b4" font-size="12">y_min</text>
  <text x="{left_margin+5}" y="{y_max_line-6:.2f}" fill="#1f77b4" font-size="12">y_max</text>

  <!-- inside points -->
  <polyline fill="none" stroke="#999" stroke-width="1.5" points="{pts_to_str(grey_pts)}" opacity="0.85"/>
  <!-- outside points -->
  <polyline fill="none" stroke="#d62728" stroke-width="2.0" points="{pts_to_str(red_pts)}" opacity="0.95"/>

  <!-- axes labels (minimal) -->
  <text x="{left_margin+5}" y="{top_margin+plot_h+18}" fill="#333" font-size="12">timestamp</text>
  <text x="10" y="{top_margin+15}" fill="#333" font-size="12">ego_pos_y</text>
</svg>
</body>
</html>
"""
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument("--out_dir", type=str, default="/home/zwx/driver_model/outputs/following_right_lane_filter")

    ap.add_argument("--right_y_min", type=float, default=-9.50)
    ap.add_argument("--right_y_max", type=float, default=-5.75)
    ap.add_argument("--gap_threshold_sec", type=float, default=0.20)
    ap.add_argument("--min_segment_duration_sec", type=float, default=0.50)

    ap.add_argument("--save_segment_csv", action="store_true", default=False)
    ap.add_argument("--save_html", action="store_true", default=True)
    ap.add_argument("--max_files", type=int, default=0, help="0 means all candidates")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    html_dir = os.path.join(args.out_dir, "html")
    seg_dir = os.path.join(args.out_dir, "extracted_segments")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    csv_paths = discover_following_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]

    seg_summary_rows: List[Dict[str, str]] = []
    compliant_rows: List[Dict[str, str]] = []
    total_files = 0
    files_with_violations = 0

    for fp in csv_paths:
        total_files += 1
        rel = os.path.relpath(fp, args.data_dir)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                rows = list(r)
        except Exception as e:
            print(f"[WARN] failed to read: {fp} ({e})")
            continue

        segs, series = detect_violation_segments(
            rows=rows,
            right_y_min=args.right_y_min,
            right_y_max=args.right_y_max,
            gap_threshold_sec=args.gap_threshold_sec,
            min_segment_duration_sec=args.min_segment_duration_sec,
        )

        if segs:
            files_with_violations += 1
        else:
            compliant_rows.append({"file": rel})

        for idx, seg in enumerate(segs, start=1):
            seg_id = f"seg{idx:03d}"
            seg_row = {
                "file": rel,
                "segment_id": seg_id,
                "violation_type": seg.violation_type,
                "start_ts": f"{seg.start_ts:.6f}",
                "end_ts": f"{seg.end_ts:.6f}",
                "duration_sec": f"{seg.duration_sec:.6f}",
                "start_frame": str(seg.start_frame),
                "end_frame": str(seg.end_frame),
                "min_ego_pos_y": f"{seg.min_y:.6f}",
                "max_ego_pos_y": f"{seg.max_y:.6f}",
                "violation_ratio": f"{seg.violation_ratio:.6f}",
            }
            seg_summary_rows.append(seg_row)

            if args.save_segment_csv:
                out_name = f"{_sanitize_filename(rel)}__{seg_id}.csv"
                out_path = os.path.join(seg_dir, out_name)
                export_segment_csv(rows, out_path, seg.start_ts, seg.end_ts)

        if args.save_html:
            html_name = f"{_sanitize_filename(rel)}.html"
            html_path = os.path.join(html_dir, html_name)
            make_svg_html(
                html_path=html_path,
                title=f"{rel}",
                series=series,
                right_y_min=args.right_y_min,
                right_y_max=args.right_y_max,
            )

    # Export summary CSV
    out_csv = os.path.join(args.out_dir, "segments_summary.csv")
    fieldnames = [
        "file",
        "segment_id",
        "violation_type",
        "start_ts",
        "end_ts",
        "duration_sec",
        "start_frame",
        "end_frame",
        "min_ego_pos_y",
        "max_ego_pos_y",
        "violation_ratio",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in seg_summary_rows:
            w.writerow(row)

    compliant_csv = os.path.join(args.out_dir, "compliant_files.csv")
    with open(compliant_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file"])
        w.writeheader()
        for row in sorted(compliant_rows, key=lambda x: x["file"]):
            w.writerow(row)

    # Build index.html
    index_path = os.path.join(args.out_dir, "index.html")
    # simple index table
    sorted_rows = sorted(seg_summary_rows, key=lambda x: (x["file"], x["segment_id"]))
    # To avoid huge html, only show top N links
    N = 500
    shown = sorted_rows[:N]

    # Build links
    link_lines = []
    for r in shown:
        rel_file = r["file"]
        html_file = f"{_sanitize_filename(rel_file)}.html"
        link_lines.append(
            f'<tr><td style="padding:6px 8px; border:1px solid #eee; white-space:nowrap;">{r["file"]}</td>'
            f'<td style="padding:6px 8px; border:1px solid #eee;">{r["segment_id"]}</td>'
            f'<td style="padding:6px 8px; border:1px solid #eee;">{r["violation_type"]}</td>'
            f'<td style="padding:6px 8px; border:1px solid #eee;">{float(r["duration_sec"]):.3f}s</td>'
            f'<td style="padding:6px 8px; border:1px solid #eee;"><a href="html/{html_file}" target="_blank">view</a></td></tr>'
        )
    compliant_sorted = sorted(compliant_rows, key=lambda x: x["file"])
    compliant_shown = compliant_sorted[:N]

    html_index = f"""<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Right-lane violations index</title></head>
<body style="font-family:sans-serif;">
<h2>Following trajectories right-lane check (ego_pos_y)</h2>
<div style="color:#444;margin-bottom:8px;">
  Candidate files: {total_files}<br/>
  Files with violations: {files_with_violations}<br/>
  Compliant files: {total_files - files_with_violations}<br/>
  Right lane y-range: [{args.right_y_min:.2f}, {args.right_y_max:.2f}]<br/>
  Min segment duration: {args.min_segment_duration_sec:.2f}s<br/>
  Gap threshold: {args.gap_threshold_sec:.2f}s
</div>

<div style="margin:12px 0;">
  <a href="segments_summary.csv">Download segments_summary.csv</a><br/>
  <a href="compliant_files.csv">Download compliant_files.csv</a>
</div>

<h3>Compliant trajectories (all)</h3>
<table style="border-collapse:collapse; font-size:13px; margin-bottom:16px;">
  <thead>
    <tr>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">file</th>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">html</th>
    </tr>
  </thead>
  <tbody>
    {''.join(
        f'<tr><td style="padding:6px 8px; border:1px solid #eee; white-space:nowrap;">{r["file"]}</td>'
        f'<td style="padding:6px 8px; border:1px solid #eee;"><a href="html/{_sanitize_filename(r["file"])}.html" target="_blank">view</a></td></tr>'
        for r in compliant_shown
    )}
  </tbody>
</table>

<h3>Violation segments</h3>
<table style="border-collapse:collapse; font-size:13px;">
  <thead>
    <tr>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">file</th>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">segment</th>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">type</th>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">duration</th>
      <th style="padding:6px 8px; border:1px solid #eee; background:#fafafa;">html</th>
    </tr>
  </thead>
  <tbody>
    {''.join(link_lines)}
  </tbody>
</table>
<div style="margin-top:10px;color:#666;">
  Showing first {len(compliant_shown)} compliant files and first {min(len(sorted_rows), N)} violation segments.
  Open compliant_files.csv / segments_summary.csv for full list.
</div>
</body></html>
"""
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_index)

    print(f"[OK] candidates: {total_files}, files_with_violations: {files_with_violations}")
    print(f"[OK] segments_summary.csv: {out_csv}")
    print(f"[OK] compliant_files.csv: {compliant_csv}")
    if args.save_html:
        print(f"[OK] html index: {index_path}")


if __name__ == "__main__":
    main()

