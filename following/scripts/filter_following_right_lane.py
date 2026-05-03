import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

"""
Filter following trajectories by right-lane occupancy.

``ego_pos_y`` is the vehicle lateral **center**. By default ``--ego_half_width_m 0.9`` treats
violation as **wheel** envelope: center must stay in
[right_y_min + hw, right_y_max - hw] so edges do not cross lane lines (straight road).
Use ``--ego_half_width_m 0`` for center-only (legacy).

With ``--save_figures`` (default on), writes Matplotlib PNGs under ``out_dir/figures/`` in the
same style as ``overtaking/scripts/segment_overtaking_phases`` (``ego_pos_y`` + ``steer`` vs
``time - t0``, right-lane band, optional center-corridor lines, violation highlights). No HTML.

python3 following/scripts/filter_following_right_lane.py \
  --data_dir /home/zwx/driver_model/following/outputs/following_calibrated \
  --out_dir /home/zwx/driver_model/following/outputs/following_right_lane_filter2
"""

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
    ego_half_width_m: float = 0.0,
) -> Tuple[List[Segment], List[Tuple[float, float, bool, float]]]:
    """
    Returns:
      - violation segments (continuous outside-range intervals)
      - point series for visualization: (timestamp, ego_pos_y, is_outside, steer)

    ``ego_pos_y`` is treated as the vehicle lateral reference **center** (simulator body center).
    If ``ego_half_width_m`` > 0, violation uses wheel/track envelope: outer edge ``y - hw`` must
    stay >= ``right_y_min``, inner edge ``y + hw`` must stay <= ``right_y_max`` (straight-road
    approximation; ignores yaw widening). If ``ego_half_width_m`` <= 0, falls back to center-only.
    """
    ts_y_outside: List[Tuple[float, float, bool, float]] = []
    segs: List[Segment] = []

    # Build time series (skip rows with missing fields)
    series: List[Tuple[float, float, int, float]] = []
    for r in rows:
        ts = parse_float(r, "timestamp")
        y = parse_float(r, "ego_pos_y")
        if ts is None or y is None:
            continue
        sv = parse_float(r, "steer")
        series.append(
            (ts, y, parse_int(r, "frame", default=0), 0.0 if sv is None else float(sv))
        )

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
    cur_has_outer = False  # outer wheel / shoulder side (y too negative)
    cur_has_inner = False  # inner wheel / toward adjacent lane (y too positive)

    hw = float(ego_half_width_m)

    def lateral_outside(y: float) -> bool:
        if hw <= 0.0:
            return (y < right_y_min) or (y > right_y_max)
        return (y - hw < right_y_min) or (y + hw > right_y_max)

    def accumulate_edge_flags(y: float) -> None:
        nonlocal cur_has_outer, cur_has_inner
        if hw <= 0.0:
            cur_has_outer = cur_has_outer or (y < right_y_min)
            cur_has_inner = cur_has_inner or (y > right_y_max)
        else:
            cur_has_outer = cur_has_outer or ((y - hw) < right_y_min)
            cur_has_inner = cur_has_inner or ((y + hw) > right_y_max)

    def violation_type_from_flags() -> str:
        if cur_has_inner and cur_has_outer:
            return "out_of_bounds"
        if cur_has_inner:
            return "too_right"
        return "too_left"

    for i, (ts, y, frame, steer) in enumerate(series):
        out = lateral_outside(y)
        ts_y_outside.append((ts, y, out, steer))

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
                cur_has_outer = False
                cur_has_inner = False
                accumulate_edge_flags(y)
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
                    cur_has_outer = False
                    cur_has_inner = False
                    accumulate_edge_flags(y)
                else:
                    cur_violation_count += 1
                    cur_total_count += 1
                    cur_min_y = min(cur_min_y, y)
                    cur_max_y = max(cur_max_y, y)
                    accumulate_edge_flags(y)
        else:
            # if we were in a segment, close it
            if cur_start_idx is not None:
                # close with end at previous index (inside row belongs outside? no, so end is i-1)
                end_ts = series[i - 1][0]
                end_frame = series[i - 1][2]
                duration = end_ts - cur_start_ts
                if duration >= min_segment_duration_sec:
                    vtype = violation_type_from_flags()
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
            vtype = violation_type_from_flags()
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


def _figure_title_ascii(s: str) -> str:
    """ASCII-safe title for matplotlib (optional folder names in rel path)."""
    if not s:
        return "session"
    out = "".join(c if 32 <= ord(c) < 127 else "_" for c in s)
    out = "_".join(x for x in out.split("_") if x)
    return out.strip("_") or "session"


def save_following_lane_figure(
    out_png_path: str,
    title: str,
    series: List[Tuple[float, float, bool, float]],
    segs: List[Segment],
    right_y_min: float,
    right_y_max: float,
    ego_half_width_m: float,
) -> bool:
    """
    Matplotlib style aligned with overtaking segment_overtaking_phases.save_phase_figure:
    top: ego_pos_y vs time with right-lane band; bottom: steer vs time.
    """
    if not series:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[WARN] matplotlib not installed; skip figure: {out_png_path}")
        return False

    times = [p[0] for p in series]
    ys = [p[1] for p in series]
    outside = [p[2] for p in series]
    steers = [p[3] for p in series]
    n = len(times)
    t0 = times[0]
    t_rel = [t - t0 for t in times]
    hw = float(ego_half_width_m)

    fig, (ax_y, ax_st) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    ax_y.axhspan(right_y_min, right_y_max, color="#c8e6c9", alpha=0.35, label="Right lane band")
    ax_y.axhline(right_y_min, color="#2e7d32", linewidth=0.8, linestyle="--")
    ax_y.axhline(right_y_max, color="#2e7d32", linewidth=0.8, linestyle="--")
    if hw > 0.0:
        y_lo = right_y_min + hw
        y_hi = right_y_max - hw
        ax_y.axhline(y_lo, color="#9467bd", linewidth=1.0, linestyle="--", alpha=0.95, label="Center corridor (wheels inside)")
        ax_y.axhline(y_hi, color="#9467bd", linewidth=1.0, linestyle="--", alpha=0.95, label="_nolegend_")

    for seg in segs:
        a = seg.start_ts - t0
        b = seg.end_ts - t0
        if b >= a:
            ax_y.axvspan(a, b, alpha=0.10, color="#f44336")
            ax_st.axvspan(a, b, alpha=0.10, color="#f44336")

    ax_y.plot(t_rel, ys, color="0.2", linewidth=1.2, label="ego_pos_y (center)")
    viol_idx = [i for i in range(n) if outside[i]]
    if viol_idx:
        ax_y.scatter(
            [t_rel[i] for i in viol_idx],
            [ys[i] for i in viol_idx],
            s=44,
            c="crimson",
            marker="x",
            linewidths=1.5,
            zorder=5,
            label="Line cross (right lane envelope)",
        )

    ax_y.set_ylabel("ego_pos_y (m)")
    n_v = sum(1 for o in outside if o)
    ax_y.set_title(
        _figure_title_ascii(title)
        + f"   n_out={n_v}/{n}   hw={hw:.2f}m   lane=[{right_y_min:.2f},{right_y_max:.2f}]"
    )
    ax_y.grid(True, alpha=0.3)
    ax_y.legend(loc="upper right", fontsize=8, ncol=2)

    ax_st.plot(t_rel, steers, color="purple", linewidth=0.9, alpha=0.85, label="steer")
    ax_st.axhline(0.0, color="0.5", linewidth=0.6)
    ax_st.set_xlabel("time - t0 (s)")
    ax_st.set_ylabel("steer")
    ax_st.legend(loc="upper right", fontsize=8)
    ax_st.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=140)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument("--out_dir", type=str, default="/home/zwx/driver_model/outputs/following_right_lane_filter")

    ap.add_argument("--right_y_min", type=float, default=-9.50)
    ap.add_argument("--right_y_max", type=float, default=-5.75)
    ap.add_argument(
        "--ego_half_width_m",
        type=float,
        default=0.9,
        help="Lateral half-width of vehicle (m): outer edge y-hw, inner y+hw vs lane edges. "
        "Trajectory ego_pos_y is body center. Set 0 to use center-only (legacy).",
    )
    ap.add_argument("--gap_threshold_sec", type=float, default=0.20)
    ap.add_argument("--min_segment_duration_sec", type=float, default=0.50)

    ap.add_argument("--save_segment_csv", action="store_true", default=False)
    ap.add_argument(
        "--no_save_figures",
        action="store_false",
        dest="save_figures",
        help="Disable PNG figures under out_dir/figures (matplotlib).",
    )
    ap.set_defaults(save_figures=True)
    ap.add_argument("--max_files", type=int, default=0, help="0 means all candidates")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_dir = os.path.join(args.out_dir, "extracted_segments")
    os.makedirs(seg_dir, exist_ok=True)

    csv_paths = discover_following_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]

    seg_summary_rows: List[Dict[str, str]] = []
    compliant_rows: List[Dict[str, str]] = []
    total_files = 0
    files_with_violations = 0
    n_figures_written = 0

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
            ego_half_width_m=args.ego_half_width_m,
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

        if args.save_figures:
            rel_slash = rel.replace("\\", "/")
            sess = os.path.basename(os.path.dirname(fp))
            stem = _sanitize_filename(sess or "session")
            fig_dir = os.path.join(args.out_dir, "figures", os.path.dirname(rel_slash))
            out_png = os.path.join(fig_dir, f"{stem}_following_lane.png")
            ok = save_following_lane_figure(
                out_png_path=out_png,
                title=rel_slash,
                series=series,
                segs=segs,
                right_y_min=args.right_y_min,
                right_y_max=args.right_y_max,
                ego_half_width_m=args.ego_half_width_m,
            )
            if ok:
                n_figures_written += 1

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

    print(f"[OK] candidates: {total_files}, files_with_violations: {files_with_violations}")
    print(f"[OK] segments_summary.csv: {out_csv}")
    print(f"[OK] compliant_files.csv: {compliant_csv}")
    if args.save_figures:
        print(f"[OK] figures written: {n_figures_written} under {os.path.join(args.out_dir, 'figures')}")


if __name__ == "__main__":
    main()

