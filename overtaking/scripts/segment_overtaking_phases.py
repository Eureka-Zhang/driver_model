# -*- coding: utf-8 -*-
"""
Segment overtaking ``driving_data.csv`` into four phases. Parameter groups:

**Right-lane geometry (visualization + wheel-band violations)** — ``--right_y_min``, ``--right_y_max``.

**Inner-edge touch predicates (P1 / P4 boundaries)** — uses ``right_y_max`` and ``--edge_eps_m``:
  ``eff_inner = right_y_max - edge_eps_m``; **L wheel** ``y+hw >= eff_inner``; **R wheel** ``y-hw >= eff_inner``.
  These are **not** the same numbers as the P2/P3 center threshold below.

**P2/P3 boundary ``B``** — ``ego_pos_y`` (center) sustained ``>= y_p2_end``:
  * ``--p2_end_mode fixed`` + ``--y_center_p2_end`` (default **-4**), or
  * ``--p2_end_mode geometry`` → ``y_p2_end = right_y_max + ego_half_width_m`` (center when **R wheel** is on the
    inner line **without** eps; e.g. -5.75 + 0.9 = **-4.85**).

Phases: 1) **following** ``[0, i_follow_end)`` — last L-wheel inner-edge run **before** ``B``; 2) **lane_change**
``[i_follow_end, B)``; 3) **left_overtake** ``[B, i_left_end)``; 4) **return** ``[i_left_end, n)`` — ``i_left_end``
from last R-wheel inner-edge run after ``B``.

Robustness: ``--segment_smooth_window`` median on y (boundaries only); ``--edge_eps_m``; ``--center_hold_sec``.

Lateral ``y`` increases toward the left lane; **L wheel** at ``y+hw``, **R wheel** at ``y-hw``.

Discovery: ``exp[123]_o``, exclude ``pre_familiarization`` / ``_b``.

可选写出 ``phase_01_following.csv`` … 或整文件索引 ``phase_segments_summary.csv``。

**合法四段会话**可在 ``out_dir`` 下按源目录镜像复制**未切分**的 ``driving_data.csv``，
与相位切片同级（``shutil.copy2``，默认开启；使用 ``--no_copy_source_driving_csv`` 关闭）。

默认 **``--save_figures``**：为每个实验生成 ``figures/<镜像路径>/session_phases.png``（全轨迹 ``ego_pos_y``、
车道带、四阶段底色、跟驰/左道超车段的轮迹压线点；**图内文字为英文**），并写 ``figures/index.html``。
压线判定与 ``filter_following_right_lane`` 一致：车体中心 ``ego_pos_y`` + ``ego_half_width_m``。

Example::

  python3 overtaking/scripts/segment_overtaking_phases.py \
    --data_dir overtaking/outputs/overtaking_p4_return_fix \
    --out_dir overtaking/outputs/overtaking_phase_segments_p4_return_fix \
    --right_y_min -9.30 --right_y_max -5.75 --p2_end_mode fixed --y_center_p2_end -4.6 --ego_half_width_m 0.9 \
    --segment_smooth_window 5 --edge_eps_m 0.08 --center_hold_sec 0.15 \
    --write_segment_csv \
    --no_copy_source_driving_csv

  # P2 end when center equals R wheel on inner line: right_y_max + hw
  # python3 ... --p2_end_mode geometry --right_y_max -5.75 --ego_half_width_m 0.9
"""
from __future__ import print_function

import argparse
import csv
import os
import re
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_float(row, key):
    v = row.get(key, "")
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _is_pre_familiarization(path):
    return "pre_familiarization" in path


def _discover_overtaking_csvs(data_dir):
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
        cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def _sanitize_filename(s):
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^0-9a-zA-Z._-]+", "_", s)
    return s.strip("_")


def _median_dt(times):
    if not times or len(times) < 2:
        return 0.02
    dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    dts.sort()
    return dts[len(dts) // 2]


def _moving_median(arr, window):
    """Odd window moving median; window<=1 returns a copy of arr."""
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


def _runs_true(flags):
    """Contiguous True runs as list of (start, end) inclusive indices."""
    n = len(flags)
    runs = []
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _first_sustained_center_ge(times, ys, y_thr, hold_sec):
    """
    First index i such that ys[i : i+need] are all >= y_thr with span >= hold_sec in time.
    Falls back to first single sample >= y_thr if hold cannot be satisfied.
    """
    n = len(ys)
    if n == 0:
        return None
    dt = _median_dt(times)
    need = max(1, int(round(float(hold_sec) / max(dt, 1e-9))))
    if need <= n:
        for i in range(n - need + 1):
            ok = True
            for k in range(need):
                if ys[i + k] < y_thr:
                    ok = False
                    break
            if ok:
                return i
    for i in range(n):
        if ys[i] >= y_thr:
            return i
    return None


def segment_indices(
    times,
    ys,
    right_y_min,
    right_y_max,
    y_center_p2_end,
    ego_half_width_m,
    segment_smooth_window=5,
    edge_eps_m=0.08,
    center_hold_sec=0.15,
):
    """
    Wheel-touch uses ``eff_inner = right_y_max - edge_eps_m`` (inner-edge band).
    Boundary ``B`` (P2 end / P3 start): smoothed center sustained ``>= y_center_p2_end`` (independent parameter).

    Slices: P1 ``[0,i_follow_end)``, P2 ``[i_follow_end,B)``, P3 ``[B,i_left_end)``, P4 ``[i_left_end,n)``.
    """
    n = len(ys)
    ref_y = float(y_center_p2_end)
    hw = float(ego_half_width_m)
    eps = float(edge_eps_m)

    if segment_smooth_window and int(segment_smooth_window) > 1:
        ys_s = _moving_median(ys, int(segment_smooth_window))
    else:
        ys_s = list(ys)

    # Inner edge of right lane (toward left lane): right_y_max (e.g. -5.75); widen toward lane interior by eps.
    eff_inner = right_y_max - eps

    if hw > 0.0:

        def left_wheel_touch_rl_inner(y):
            return (y + hw) >= eff_inner

        def right_wheel_touch_rl_inner(y):
            return (y - hw) <= eff_inner
    else:

        def left_wheel_touch_rl_inner(y):
            return y >= eff_inner

        def right_wheel_touch_rl_inner(y):
            return y >= eff_inner

    if not times or len(times) != n:
        times = [float(i) for i in range(n)]

    B = _first_sustained_center_ge(times, ys_s, y_center_p2_end, center_hold_sec)

    if B is None:
        return {
            "i_follow_end": n,
            "i_lc_end": n,
            "i_left_end": n,
            "i_reach": None,
            "y_p2_end": ref_y,
            "status": "never_center_y_lane_change",
        }

    flags_L = [left_wheel_touch_rl_inner(ys_s[i]) for i in range(B)]
    runs_L = _runs_true(flags_L)
    if runs_L:
        i_follow_end = runs_L[-1][0]
    else:
        i_follow_end = B

    if i_follow_end > B:
        i_follow_end = B

    tail = [right_wheel_touch_rl_inner(ys_s[i]) for i in range(B, n)]
    runs_R = _runs_true(tail)
    if runs_R:
        i_left_end = B + runs_R[-1][0]
    else:
        i_left_end = n

    if i_left_end >= n:
        return {
            "i_follow_end": i_follow_end,
            "i_lc_end": B,
            "i_left_end": n,
            "i_reach": B,
            "y_p2_end": ref_y,
            "status": "no_r_wheel_rl_inner_edge",
        }

    return {
        "i_follow_end": i_follow_end,
        "i_lc_end": B,
        "i_left_end": i_left_end,
        "i_reach": B,
        "y_p2_end": ref_y,
        "status": "ok",
    }


def four_phases_all_non_empty(out, n):
    """
    Require ``status == 'ok'`` and **no zero-length phase** among P1–P4.

    Phases: ``P1=[0,i1), P2=[i1,B), P3=[B,i3), P4=[i3,n)``. Lengths are
    ``len(P1)=i1``, ``len(P2)=B-i1``, ``len(P3)=i3-B``, ``len(P4)=n-i3``;
    all must be ``>= 1``, i.e. ``0 < i1 < B < i3 < n``.

    Returns ``(ok, detail)`` where ``detail`` is empty if ok, else a short reason for logging.
    """
    st = out.get("status")
    i1 = out.get("i_follow_end")
    i_reach = out.get("i_reach")
    i3 = out.get("i_left_end")
    if st == "never_center_y_lane_change":
        return False, "status=never_center_y_lane_change (no P2/P3/P4 boundary B)"
    if st == "no_r_wheel_rl_inner_edge":
        return False, "status=no_r_wheel_rl_inner_edge (no P4 boundary)"
    if st != "ok":
        return False, "status={}".format(st)
    if i_reach is None:
        return False, "i_reach is None"
    try:
        i1, i_reach, i3 = int(i1), int(i_reach), int(i3)
    except (TypeError, ValueError):
        return False, "non-int boundaries i1={!r} B={!r} i3={!r}".format(i1, i_reach, i3)
    lp1, lp2, lp3, lp4 = i1, i_reach - i1, i3 - i_reach, n - i3
    if lp1 < 1 or lp2 < 1 or lp3 < 1 or lp4 < 1:
        return False, (
            "zero-length phase not allowed: n={} i1={} B={} i3={} "
            "len_P1={} len_P2={} len_P3={} len_P4={}".format(
                n, i1, i_reach, i3, lp1, lp2, lp3, lp4
            )
        )
    return True, ""


def _slice_rows(rows, i0, i1):
    if i0 >= i1:
        return []
    return rows[i0:i1]


def _wheel_violation_right(y, rmin, rmax, hw):
    """True if reference center y implies wheel outside right lane band."""
    if hw <= 0.0:
        return (y < rmin) or (y > rmax)
    return ((y - hw) < rmin) or ((y + hw) > rmax)


def _wheel_violation_left(y, lmin, lmax, hw):
    """True if wheel outside left lane band."""
    if hw <= 0.0:
        return (y < lmin) or (y > lmax)
    return ((y - hw) < lmin) or ((y + hw) > lmax)


def _collect_violation_indices(ys, idx_range, checker):
    """idx_range is (start, end) exclusive end; checker(y) -> bool."""
    a, b = idx_range
    out = []
    for i in range(max(0, a), min(len(ys), b)):
        if checker(ys[i]):
            out.append(i)
    return out


def _figure_title_ascii(s):
    """ASCII-safe title for matplotlib (path rel may contain non-Latin folder names)."""
    if not s:
        return "session"
    out = "".join(c if 32 <= ord(c) < 127 else "_" for c in s)
    out = "_".join(x for x in out.split("_") if x)
    return out.strip("_") or "session"


def save_phase_figure(
    out_png_path,
    title,
    times,
    ys,
    steers,
    i_follow_end,
    i_reach,
    i_left_end,
    n,
    status,
    right_y_min,
    right_y_max,
    left_y_min,
    left_y_max,
    ego_half_width_m,
    y_center_p2_end,
):
    """Full-session figure: lane bands, phase shading, line-crossing in phase1 & phase3."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("[WARN] matplotlib not installed; skip figure:", out_png_path)
        return False

    t0 = times[0]
    t_rel = [ti - t0 for ti in times]
    hw = float(ego_half_width_m)

    def chk_r(y):
        return _wheel_violation_right(y, right_y_min, right_y_max, hw)

    def chk_l(y):
        return _wheel_violation_left(y, left_y_min, left_y_max, hw)

    # Right-lane envelope violations: split by phase (was only P1; lane-change can still violate right band).
    viol_r_p1_idx = _collect_violation_indices(ys, (0, i_follow_end), chk_r) if i_follow_end > 0 else []
    viol_r_p2_idx = []
    if i_reach is not None and i_reach > i_follow_end:
        viol_r_p2_idx = _collect_violation_indices(ys, (i_follow_end, i_reach), chk_r)

    # Left-lane envelope violations: P3 left OT + P4 return (was only P3; return can still be wide in left lane).
    viol_l_p3_idx = []
    viol_l_p4_idx = []
    if i_reach is not None and i_left_end is not None and i_reach < n:
        if i_left_end > i_reach:
            viol_l_p3_idx = _collect_violation_indices(ys, (i_reach, i_left_end), chk_l)
        if i_left_end < n:
            viol_l_p4_idx = _collect_violation_indices(ys, (i_left_end, n), chk_l)

    fig, (ax_y, ax_st) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    ax_y.axhspan(right_y_min, right_y_max, color="#c8e6c9", alpha=0.35, label="Right lane band")
    ax_y.axhspan(left_y_min, left_y_max, color="#bbdefb", alpha=0.35, label="Left lane band")
    ax_y.axhline(right_y_min, color="#2e7d32", linewidth=0.8, linestyle="--")
    ax_y.axhline(right_y_max, color="#2e7d32", linewidth=0.8, linestyle="--")
    ax_y.axhline(left_y_min, color="#1565c0", linewidth=0.8, linestyle="--")
    ax_y.axhline(left_y_max, color="#1565c0", linewidth=0.8, linestyle="--")
    if hw > 0.0:
        ax_y.axhline(
            right_y_max - hw,
            color="#6d4c41",
            linewidth=0.9,
            linestyle=":",
            alpha=0.9,
            label="center if L wheel on RL inner edge",
        )
        ax_y.axhline(
            right_y_max + hw,
            color="#6a1b9a",
            linewidth=0.9,
            linestyle=":",
            alpha=0.9,
            label="center if R wheel on RL inner edge",
        )
    ax_y.axhline(
        y_center_p2_end,
        color="#e65100",
        linewidth=0.9,
        linestyle=":",
        alpha=0.9,
        label="P2 end threshold (center >=)",
    )

    span_alpha = 0.26

    def span(ax, i0, i1, color):
        if i0 >= i1 or i1 > n or i0 < 0:
            return
        ax.axvspan(t_rel[i0], t_rel[min(i1, n) - 1], alpha=span_alpha, color=color)

    phase_patches = (
        Patch(facecolor="#4caf50", alpha=0.4, edgecolor="#1b5e20", linewidth=0.6, label="P1 following"),
        Patch(facecolor="#ffc107", alpha=0.4, edgecolor="#f57f17", linewidth=0.6, label="P2 lane change"),
        Patch(facecolor="#2196f3", alpha=0.4, edgecolor="#0d47a1", linewidth=0.6, label="P3 left overtake"),
        Patch(facecolor="#9e9e9e", alpha=0.4, edgecolor="#424242", linewidth=0.6, label="P4 return"),
    )

    span(ax_y, 0, i_follow_end, "#4caf50")
    if i_reach is not None:
        span(ax_y, i_follow_end, i_reach, "#ffc107")
        span(ax_y, i_reach, i_left_end, "#2196f3")
        span(ax_y, i_left_end, n, "#9e9e9e")
    else:
        # Never reached left lane center: post-following = lane-change / incomplete
        span(ax_y, i_follow_end, n, "#ffc107")

    ax_y.plot(t_rel, ys, color="0.2", linewidth=1.2, label="ego_pos_y (center)")

    if viol_r_p1_idx:
        ax_y.scatter(
            [t_rel[i] for i in viol_r_p1_idx],
            [ys[i] for i in viol_r_p1_idx],
            s=44,
            c="crimson",
            marker="x",
            linewidths=1.5,
            zorder=6,
            label="Right lane cross (P1 following)",
        )
    if viol_r_p2_idx:
        ax_y.scatter(
            [t_rel[i] for i in viol_r_p2_idx],
            [ys[i] for i in viol_r_p2_idx],
            s=46,
            c="darkorange",
            marker="x",
            linewidths=1.6,
            zorder=6,
            label="Right lane cross (P2 lane change)",
        )
    if viol_l_p3_idx:
        ax_y.scatter(
            [t_rel[i] for i in viol_l_p3_idx],
            [ys[i] for i in viol_l_p3_idx],
            s=42,
            facecolors="none",
            edgecolors="darkorange",
            linewidths=1.8,
            marker="o",
            zorder=6,
            label="Left lane cross (P3 left OT)",
        )
    if viol_l_p4_idx:
        ax_y.scatter(
            [t_rel[i] for i in viol_l_p4_idx],
            [ys[i] for i in viol_l_p4_idx],
            s=44,
            facecolors="none",
            edgecolors="purple",
            linewidths=1.8,
            marker="o",
            zorder=6,
            label="Left lane cross (P4 return)",
        )

    for idx, c in ((i_follow_end, "green"), (i_reach, "blue"), (i_left_end, "gray")):
        if idx is not None and 0 < idx < n:
            ax_y.axvline(t_rel[idx], color=c, linestyle=":", linewidth=1.0, alpha=0.85)

    ax_y.set_ylabel("ego_pos_y (m)")
    ax_y.set_title(_figure_title_ascii(title) + "   status=%s   hw=%.2fm" % (status, hw))
    ax_y.grid(True, alpha=0.3)
    h_y, lab_y = ax_y.get_legend_handles_labels()
    ax_y.legend(
        handles=list(phase_patches) + h_y,
        labels=[p.get_label() for p in phase_patches] + lab_y,
        loc="upper right",
        fontsize=7,
        ncol=2,
    )

    ax_st.plot(t_rel, steers, color="purple", linewidth=0.9, alpha=0.85, label="steer")
    ax_st.axhline(0.0, color="0.5", linewidth=0.6)
    span(ax_st, 0, i_follow_end, "#4caf50")
    if i_reach is not None:
        span(ax_st, i_follow_end, i_reach, "#ffc107")
        span(ax_st, i_reach, i_left_end, "#2196f3")
        span(ax_st, i_left_end, n, "#9e9e9e")
    else:
        span(ax_st, i_follow_end, n, "#ffc107")
    ax_st.set_xlabel("time - t0 (s)")
    ax_st.set_ylabel("steer")
    h_s, lab_s = ax_st.get_legend_handles_labels()
    ax_st.legend(
        handles=list(phase_patches) + h_s,
        labels=[p.get_label() for p in phase_patches] + lab_s,
        loc="upper right",
        fontsize=7,
        ncol=2,
    )
    ax_st.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=140)
    plt.close(fig)
    return True


def main():
    _repo = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
    ap = argparse.ArgumentParser(
        description="Segment overtaking CSV into following / lane_change / left_overtake / return phases.",
    )
    ap.add_argument("--data_dir", type=str, default=os.path.join(_repo, "data"))
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_repo, "overtaking", "outputs", "overtaking_phase_segments"),
    )
    ap.add_argument("--left_y_min", type=float, default=-5.55)
    ap.add_argument("--left_y_max", type=float, default=-2.20)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument(
        "--right_y_max",
        type=float,
        default=-5.75,
        help="Right lane inner edge (m): inner-edge touch uses right_y_max - edge_eps_m (not the P2 center threshold).",
    )
    ap.add_argument(
        "--p2_end_mode",
        choices=("fixed", "geometry"),
        default="fixed",
        help=(
            "P2/P3 boundary B: 'fixed' uses --y_center_p2_end (default -4). "
            "'geometry' uses right_y_max + ego_half_width_m (R wheel on inner line without eps)."
        ),
    )
    ap.add_argument(
        "--y_center_p2_end",
        type=float,
        default=None,
        help=(
            "With --p2_end_mode fixed: smoothed ego_pos_y sustains >= this to end P2 / start P3. "
            "Default -4 when unset; ignored when p2_end_mode=geometry."
        ),
    )
    ap.add_argument(
        "--y_center_lane_change",
        type=float,
        default=None,
        help="Deprecated: same meaning as --y_center_p2_end. Do not pass both.",
    )
    ap.add_argument(
        "--segment_smooth_window",
        type=int,
        default=5,
        help="Odd window length for moving median on ego_pos_y when detecting boundaries only; 1 disables.",
    )
    ap.add_argument(
        "--edge_eps_m",
        type=float,
        default=0.08,
        help="Tolerance (m): inner-edge threshold uses right_y_max - eps for wheel-touch predicates.",
    )
    ap.add_argument(
        "--center_hold_sec",
        type=float,
        default=0.15,
        help="Smoothed center must stay >= resolved P2 threshold for this long to mark B; else first sample >= threshold.",
    )
    ap.add_argument("--write_segment_csv", action="store_true", default=False)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument(
        "--ego_half_width_m",
        type=float,
        default=0.9,
        help="Lateral half-width (m): L wheel at y+hw, R wheel at y-hw; also used for phase boundary rules.",
    )
    ap.add_argument(
        "--no_save_figures",
        action="store_false",
        dest="save_figures",
        help="Disable PNG figures and figures/index.html.",
    )
    ap.add_argument(
        "--no_copy_source_driving_csv",
        action="store_false",
        dest="copy_source_driving_csv",
        help=(
            "Do not copy the source driving_data.csv into out_dir/<mirrored>/ for valid four-phase sessions "
            "(default: copy with shutil.copy2)."
        ),
    )
    ap.set_defaults(save_figures=True, copy_source_driving_csv=True)
    args = ap.parse_args()

    if args.y_center_p2_end is not None and args.y_center_lane_change is not None:
        ap.error("Use only one of --y_center_p2_end and --y_center_lane_change.")

    if args.p2_end_mode == "geometry":
        y_p2_resolved = float(args.right_y_max) + float(args.ego_half_width_m)
    elif args.y_center_p2_end is not None:
        y_p2_resolved = float(args.y_center_p2_end)
    elif args.y_center_lane_change is not None:
        y_p2_resolved = float(args.y_center_lane_change)
    else:
        y_p2_resolved = -4.5

    csv_paths = _discover_overtaking_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        csv_paths = csv_paths[: args.max_files]

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    figure_links = []
    n_skipped_incomplete = 0

    for fp in csv_paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            raw = list(reader)
        if not fieldnames or not raw:
            continue

        rows_ts = []
        for row in raw:
            ts = _parse_float(row, "timestamp")
            y = _parse_float(row, "ego_pos_y")
            if ts is None or y is None:
                continue
            rows_ts.append((ts, y, row))
        rows_ts.sort(key=lambda x: x[0])
        if not rows_ts:
            continue

        times = [x[0] for x in rows_ts]
        ys = [x[1] for x in rows_ts]
        rows = [x[2] for x in rows_ts]
        n = len(times)
        steers = [_parse_float(r, "steer") for r in rows]
        steers = [0.0 if s is None else s for s in steers]

        out = segment_indices(
            times,
            ys,
            args.right_y_min,
            args.right_y_max,
            y_p2_resolved,
            args.ego_half_width_m,
            segment_smooth_window=args.segment_smooth_window,
            edge_eps_m=args.edge_eps_m,
            center_hold_sec=args.center_hold_sec,
        )

        ok_four, four_detail = four_phases_all_non_empty(out, n)
        if not ok_four:
            n_skipped_incomplete += 1
            print(
                "[SKIP] 未检出完整四相位（已丢弃）{}  |  {}".format(
                    rel.replace("/driving_data.csv", "") or rel,
                    four_detail,
                )
            )
            continue

        i1 = out["i_follow_end"]
        i_reach = out.get("i_reach")
        i3 = out["i_left_end"]
        lcy = float(out["y_p2_end"])
        st = out["status"]

        hw = float(args.ego_half_width_m)
        viol_n_follow = (
            len(
                _collect_violation_indices(
                    ys,
                    (0, i1),
                    lambda y, rmin=args.right_y_min, rmax=args.right_y_max, h=hw: _wheel_violation_right(
                        y, rmin, rmax, h
                    ),
                )
            )
            if i1 > 0
            else 0
        )
        viol_n_left = 0
        if i_reach is not None and i3 is not None and i_reach < n:
            viol_n_left = len(
                _collect_violation_indices(
                    ys,
                    (i_reach, i3),
                    lambda y, lmin=args.left_y_min, lmax=args.left_y_max, h=hw: _wheel_violation_left(
                        y, lmin, lmax, h
                    ),
                )
            )

        t_start = times[0]
        t_end = times[-1]
        if i_reach is not None and i_reach < n:
            t_reach = times[i_reach]
        else:
            t_reach = t_end
        t_follow_end = times[i1 - 1] if i1 > 0 else t_start
        if i3 > 0 and i3 <= n:
            t_left_end = times[i3 - 1]
        else:
            t_left_end = t_end

        summary.append(
            {
                "file": rel,
                "status": st,
                "p2_end_mode": args.p2_end_mode,
                "y_p2_end": "{:.6f}".format(lcy),
                "n_rows": str(n),
                "i_end_following": str(i1),
                "i_p2_end": str(i_reach if i_reach is not None else ""),
                "i_end_left_overtake": str(i3),
                "t0": "{:.6f}".format(t_start),
                "t_end_following": "{:.6f}".format(t_follow_end),
                "t_p2_end": "{:.6f}".format(t_reach),
                "t_end_left_overtake": "{:.6f}".format(t_left_end),
                "t_end": "{:.6f}".format(t_end),
                "viol_n_following": str(viol_n_follow),
                "viol_n_left_overtake": str(viol_n_left),
            }
        )

        session_out_dir = os.path.join(args.out_dir, os.path.dirname(rel))
        need_session_out_dir = bool(args.copy_source_driving_csv) or (
            args.write_segment_csv and i_reach is not None and st == "ok"
        )
        if need_session_out_dir:
            os.makedirs(session_out_dir, exist_ok=True)
        if args.copy_source_driving_csv:
            shutil.copy2(fp, os.path.join(session_out_dir, "driving_data.csv"))

        if args.save_figures:
            sess = os.path.basename(os.path.dirname(fp))
            stem = _sanitize_filename(sess or "session")
            fig_dir = os.path.join(args.out_dir, "figures", os.path.dirname(rel))
            out_png = os.path.join(fig_dir, "{}_phases.png".format(stem))
            title_bits = rel.replace("/driving_data.csv", "").replace("driving_data.csv", "")
            ok = save_phase_figure(
                out_png,
                title_bits or rel,
                times,
                ys,
                steers,
                i1,
                i_reach,
                i3,
                n,
                st,
                args.right_y_min,
                args.right_y_max,
                args.left_y_min,
                args.left_y_max,
                args.ego_half_width_m,
                y_p2_resolved,
            )
            if ok:
                idx_root = os.path.join(args.out_dir, "figures")
                href = os.path.relpath(out_png, idx_root).replace("\\", "/")
                figure_links.append((href, rel))

        if args.write_segment_csv and i_reach is not None and st == "ok":
            base = session_out_dir
            sess = os.path.basename(os.path.dirname(fp))
            stem = _sanitize_filename(sess or "session")
            segs = [
                ("phase_01_following", 0, i1),
                ("phase_02_lane_change", i1, i_reach),
                ("phase_03_left_overtake", i_reach, i3),
                ("phase_04_return", i3, n),
            ]
            for name, a, b in segs:
                part = _slice_rows(rows, a, b)
                out_fp = os.path.join(base, "{}__{}.csv".format(stem, name))
                if not part:
                    continue
                with open(out_fp, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fieldnames)
                    w.writeheader()
                    for r in part:
                        w.writerow(r)

    sum_fp = os.path.join(args.out_dir, "phase_segments_summary.csv")
    with open(sum_fp, "w", newline="", encoding="utf-8") as f:
        keys = [
            "file",
            "status",
            "p2_end_mode",
            "y_p2_end",
            "n_rows",
            "i_end_following",
            "i_p2_end",
            "i_end_left_overtake",
            "t0",
            "t_end_following",
            "t_p2_end",
            "t_end_left_overtake",
            "t_end",
            "viol_n_following",
            "viol_n_left_overtake",
        ]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary:
            w.writerow(row)

    if args.save_figures and figure_links:
        idx_path = os.path.join(args.out_dir, "figures", "index.html")
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        lines = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN"><head><meta charset="utf-8"/>',
            "<title>Overtaking phase figures</title>",
            "</head><body>",
            "<h1>Overtaking phase figures</h1>",
            "<ul>",
        ]
        for href, label in sorted(figure_links, key=lambda x: x[1]):
            lines.append('  <li><a href="%s">%s</a></li>' % (href, label))
        lines.extend(["</ul>", "</body></html>"])
        with open(idx_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

    print("[OK] files:", len(summary), "  skipped_incomplete_four_phases:", n_skipped_incomplete)
    print("[OK] summary:", sum_fp)
    if args.save_figures and figure_links:
        print("[OK] figures index:", os.path.join(args.out_dir, "figures", "index.html"))


if __name__ == "__main__":
    main()
