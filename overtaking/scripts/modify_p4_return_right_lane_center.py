# -*- coding: utf-8 -*-
"""
Modify overtaking ``driving_data.csv`` **P4 return** lateral profile (``ego_pos_y`` only),
with an extra safety check on the **right wheel** after returning to the **right-lane center**.

New rule (per user request)
---------------------------
In P4 (return to right lane):

- If the trajectory **reaches the right-lane center** (default ``-7.625``) at least once
  (first sustained arrival), then on the time span from that first arrival and onwards:
  - Check if the **right wheel** touches/presses the **right lane outer edge** (default ``-9.30``).
  - If it does, apply the **minimal upward correction** on ``ego_pos_y`` in that span so that
    the right wheel no longer presses the edge.
- If it **never reaches** the right-lane center in P4: **do not modify anything**.

Right wheel y-position is approximated as ``ego_pos_y - ego_half_width_m``.

Rows are sorted by ``timestamp``. Other columns unchanged except ``ego_pos_y`` where modified.

Discovery matches ``segment_overtaking_phases``: ``exp[123]_o``, exclude ``pre_familiarization`` / ``_b``.

Example::

  python3 overtaking/scripts/modify_p4_return_right_lane_center.py \
    --data_dir overtaking/outputs/selected_p1p3_lateral \
    --out_dir overtaking/outputs/overtaking_p4_return_fix \
    --right_lane_center -7.625 \
    --right_edge_y -9.30 \
    --center_tol_m 0.15 --center_hold_sec 0.2 \
    --smooth_window 11
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import segment_overtaking_phases as ov_seg


def _parse_float_any(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _moving_average(vals: Sequence[float], window: int) -> List[float]:
    if window <= 1 or not vals:
        return list(vals)
    w = int(window) | 1
    half = w // 2
    n = len(vals)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(vals[lo:hi]) / float(hi - lo))
    return out


def _moving_median(vals: Sequence[float], window: int) -> List[float]:
    if window <= 1 or not vals:
        return list(vals)
    w = int(window) | 1
    half = w // 2
    n = len(vals)
    out: List[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(float(x) for x in vals[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def _smoothstep(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def _contiguous_runs(mask: Sequence[bool], base_idx: int) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        runs.append((base_idx + i, base_idx + j))
        i = j + 1
    return runs


def _merge_runs_nearby(runs: List[Tuple[int, int]], merge_gap_frames: int) -> List[List[int]]:
    if not runs:
        return []
    runs = sorted(runs)
    merged = [[int(runs[0][0]), int(runs[0][1])]]
    for s0, e0 in runs[1:]:
        ps, pe = merged[-1]
        if int(s0) - pe - 1 <= int(merge_gap_frames):
            merged[-1][1] = max(pe, int(e0))
        else:
            merged.append([int(s0), int(e0)])
    return merged


def _expand_interval(s: int, e: int, pad: int, lo: int, hi: int) -> Tuple[int, int]:
    S = max(int(lo), int(s) - int(pad))
    E = min(int(hi) - 1, int(e) + int(pad))
    return S, E


def _build_adjusted_span_shift(
    ys_raw: Sequence[float],
    S: int,
    E: int,
    s: int,
    e: int,
    shift: float,
) -> List[float]:
    """Smoothstep ramps of additive shift in pads; full shift on [s,e]."""
    out: List[float] = []
    sh = float(shift)
    for i in range(int(S), int(E) + 1):
        y0 = float(ys_raw[i])
        if i < s:
            nleft = s - S
            if nleft <= 0:
                y = y0 + sh
            else:
                u = float(i - S + 1) / float(nleft + 1)
                w = _smoothstep(u)
                y = y0 + w * sh
        elif i > e:
            nright = E - e
            if nright <= 0:
                y = y0 + sh
            else:
                u = float(E - i + 1) / float(nright + 1)
                w = _smoothstep(u)
                y = y0 + w * sh
        else:
            y = y0 + sh
        out.append(y)
    return out


def _cosine_span_weight(k: int, span_len: int, taper_frames: int) -> float:
    if taper_frames <= 0:
        return 1.0
    d = min(int(k), int(span_len) - 1 - int(k))
    if d >= int(taper_frames):
        return 1.0
    u = float(d) / float(taper_frames)
    return 0.5 * (1.0 - math.cos(math.pi * u))


def _smooth_changed_segments_ma(
    y: List[float],
    ys_raw: Sequence[float],
    changed: List[bool],
    window: int,
) -> None:
    """MA each contiguous changed run; pad with raw neighbors when possible."""
    w_in = int(window)
    if w_in <= 1 or not any(changed):
        return
    w = int(w_in) | 1
    half = w // 2
    n = len(y)
    i = 0
    while i < n:
        if not changed[i]:
            i += 1
            continue
        lo = i
        while i + 1 < n and changed[i + 1]:
            i += 1
        hi = i
        seg = [float(y[j]) for j in range(lo, hi + 1)]
        left_pad_rev: List[float] = []
        for t in range(1, half + 1):
            idx = lo - t
            if idx >= 0 and not changed[idx]:
                left_pad_rev.append(float(ys_raw[idx]))
            else:
                left_pad_rev.append(seg[0])
        left_pad = list(reversed(left_pad_rev))
        right_pad: List[float] = []
        for t in range(1, half + 1):
            idx = hi + t
            if idx < n and not changed[idx]:
                right_pad.append(float(ys_raw[idx]))
            else:
                right_pad.append(seg[-1])
        padded = left_pad + seg + right_pad
        sm = _moving_average(padded, w)
        for j in range(lo, hi + 1):
            y[j] = sm[half + (j - lo)]
        i = hi + 1


def _junction_blend_into_raw_neighbors(
    ys_out: List[float],
    ys_raw: Sequence[float],
    changed: List[bool],
    lo_limit: int,
    hi_limit: int,
    blend_frames: int,
) -> None:
    """Blend a few raw neighbors toward each repaired run edge (smoothstep) within [lo_limit,hi_limit)."""
    jb = max(0, int(blend_frames))
    if jb <= 0:
        return
    n = len(ys_out)
    lo_limit = max(0, int(lo_limit))
    hi_limit = min(int(hi_limit), n)
    eligible_before = list(changed)
    runs: List[Tuple[int, int, float, float]] = []
    i = lo_limit
    while i < hi_limit:
        if not eligible_before[i]:
            i += 1
            continue
        lo = i
        while i + 1 < hi_limit and eligible_before[i + 1]:
            i += 1
        hi = i
        runs.append((lo, hi, float(ys_out[lo]), float(ys_out[hi])))
        i = hi + 1
    if not runs:
        return
    scale = float(jb + 1)
    for lo, hi, ref_l, ref_r in runs:
        for u in range(1, jb + 1):
            idx = lo - u
            if idx >= lo_limit and not eligible_before[idx]:
                wx = _smoothstep(1.0 - float(u) / scale)
                ys_out[idx] = (1.0 - wx) * float(ys_raw[idx]) + wx * ref_l
                changed[idx] = True
        for u in range(1, jb + 1):
            idx = hi + u
            if idx < hi_limit and not eligible_before[idx]:
                wx = _smoothstep(1.0 - float(u) / scale)
                ys_out[idx] = (1.0 - wx) * float(ys_raw[idx]) + wx * ref_r
                changed[idx] = True


def _smooth_y_span(
    y: List[float],
    lo: int,
    hi_excl: int,
    window: int,
) -> None:
    """In-place MA on ``y[lo:hi_excl]`` with replicate pad at corridor edges."""
    w_in = int(window)
    if w_in <= 1 or hi_excl <= lo:
        return
    w = int(w_in) | 1
    half = w // 2
    seg = [float(y[i]) for i in range(lo, hi_excl)]
    pad_l = float(y[lo - 1]) if lo > 0 else float(seg[0])
    pad_r = float(y[hi_excl]) if hi_excl < len(y) else float(seg[-1])
    padded = ([pad_l] * half) + seg + ([pad_r] * half)
    sm = _moving_average(padded, w)
    for k, idx in enumerate(range(lo, hi_excl)):
        y[idx] = sm[half + k]


def _right_wheel_press_edge(y_center: float, ego_half_width_m: float, right_edge_y: float) -> bool:
    """True when right wheel is at/beyond (<=) the right edge line."""
    y_rw = float(y_center) - float(ego_half_width_m)
    return y_rw <= float(right_edge_y)


def _apply_right_edge_fix(
    ys: List[float],
    lo: int,
    hi_excl: int,
    *,
    ego_half_width_m: float,
    right_edge_y: float,
    margin_m: float = 0.01,
    repair_inward_m: float = 0.02,
    merge_gap_frames: int = 3,
    pad_frames: int = 18,
    median_window: int = 11,
    ma_window: int = 13,
    span_edge_taper: int = 15,
) -> Tuple[int, float]:
    """
    Reference behaviour: similar to ``smooth_selected_p1_p3_lateral.py``.

    - Detect contiguous right-edge violation runs in [lo,hi_excl)
    - For each run, compute minimal upward shift to clear the edge (plus ``repair_inward_m``)
    - Apply shift with smoothstep ramps in pad frames
    - Median + MA smooth the adjusted span and cosine-taper mix at span edges

    Returns (n_spans_repaired, max_shift_applied).
    """
    hw = float(ego_half_width_m)
    edge = float(right_edge_y)
    margin = abs(float(margin_m))
    inward = max(0.0, float(repair_inward_m))

    n = len(ys)
    lo_i = max(0, int(lo))
    hi_i = min(int(hi_excl), n)
    if lo_i >= hi_i:
        return 0, 0.0

    def chk(y_center: float) -> bool:
        return (float(y_center) - hw) <= (edge + margin)

    mask = [chk(float(ys[i])) for i in range(lo_i, hi_i)]
    runs = _merge_runs_nearby(_contiguous_runs(mask, lo_i), int(merge_gap_frames))
    if not runs:
        return 0, 0.0

    ys_raw = list(ys)
    ys_out = list(ys)
    changed = [False] * n

    n_spans = 0
    max_shift = 0.0
    for s0, e0 in runs:
        s0 = max(lo_i, int(s0))
        e0 = min(hi_i - 1, int(e0))
        if s0 > e0:
            continue
        # minimal upward shift needed so that (y + shift) - hw > edge + margin
        needed = 0.0
        for i in range(s0, e0 + 1):
            y = float(ys_raw[i])
            need_i = (edge + hw + margin) - y
            needed = max(needed, need_i)
        if needed <= 0.0:
            continue
        shift = needed + inward
        max_shift = max(max_shift, shift)

        S, E = _expand_interval(s0, e0, int(pad_frames), lo_i, hi_i)
        if S > E:
            continue
        buf = _build_adjusted_span_shift(ys_raw, S, E, s0, e0, shift)
        sm = _moving_median(buf, int(median_window))
        sm = _moving_average(sm, int(ma_window))

        span_len = E - S + 1
        for k in range(span_len):
            i = S + k
            w_mix = _cosine_span_weight(k, span_len, int(span_edge_taper))
            ys_out[i] = (1.0 - w_mix) * float(buf[k]) + w_mix * float(sm[k])
            changed[i] = True
        n_spans += 1

    if n_spans <= 0:
        return 0, 0.0

    # Post smooth + junction blend (local to [lo_i, hi_i))
    _smooth_changed_segments_ma(ys_out, ys_raw, changed, window=25)
    _junction_blend_into_raw_neighbors(
        ys_out, ys_raw, changed, lo_limit=lo_i, hi_limit=hi_i, blend_frames=12
    )

    # Commit back
    for i in range(lo_i, hi_i):
        if changed[i]:
            ys[i] = ys_out[i]

    return n_spans, max_shift


def _first_sustained_near_center(
    times: Sequence[float],
    ys: Sequence[float],
    ile: int,
    n: int,
    y_center: float,
    tol_m: float,
    hold_sec: float,
) -> Optional[int]:
    """First index ``i ∈ [ile,n)`` with ``|ys[i+k]-y_center|≤tol`` for ``k=0..need-1``; then fallback single hit."""
    if ile >= n:
        return None
    dt = ov_seg._median_dt(list(times))
    need = max(1, int(round(float(hold_sec) / max(dt, 1e-9))))
    need = min(need, max(1, n - ile))
    yc = float(y_center)
    tol = abs(float(tol_m))
    if ile + need <= n:
        for i in range(ile, n - need + 1):
            ok = True
            for k in range(need):
                if abs(float(ys[i + k]) - yc) > tol:
                    ok = False
                    break
            if ok:
                return i
    for i in range(ile, n):
        if abs(float(ys[i]) - yc) <= tol:
            return i
    return None


def _sorted_valid_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[float], List[float]]:
    acc: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        ts = _parse_float_any(row, "timestamp")
        y = _parse_float_any(row, "ego_pos_y")
        if ts is None or y is None:
            continue
        acc.append((ts, dict(row)))
    acc.sort(key=lambda x: x[0])
    if not acc:
        return [], [], []
    rorder = [p[1] for p in acc]
    times = [float(p[0]) for p in acc]
    ys = []
    last_y = 0.0
    ok_y = False
    for row in rorder:
        v = _parse_float_any(row, "ego_pos_y")
        if v is not None:
            last_y = float(v)
            ok_y = True
        ys.append(last_y if ok_y else 0.0)
    return rorder, times, ys


def process_file(
    in_path: str,
    out_path: str,
    *,
    ego_half_width_m: float,
    right_y_min: float,
    right_y_max: float,
    right_lane_center: float,
    right_edge_y: float,
    y_center_p2_end: Optional[float],
    p2_end_mode: str,
    segment_smooth_window: int,
    edge_eps_m: float,
    center_hold_seg_sec: float,
    center_tol_m: float,
    center_hold_sec: float,
    smooth_window: int,
) -> Dict[str, str]:
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    if not fieldnames or not rows:
        return {"status": "skip", "reason": "empty"}

    aligned, times, ys_raw = _sorted_valid_rows(rows)
    n = len(aligned)
    if n == 0 or "ego_pos_y" not in fieldnames:
        return {"status": "skip", "reason": "no_ts_y"}

    hw = float(ego_half_width_m)
    if p2_end_mode == "geometry":
        y_thr = float(right_y_max) + hw
    else:
        y_thr = float(y_center_p2_end) if y_center_p2_end is not None else -4.5

    ys_med = (
        ov_seg._moving_median(list(ys_raw), int(segment_smooth_window))
        if segment_smooth_window > 1
        else list(ys_raw)
    )
    sg = ov_seg.segment_indices(
        times,
        ys_med,
        right_y_min,
        right_y_max,
        y_thr,
        ego_half_width_m,
        segment_smooth_window=1,
        edge_eps_m=edge_eps_m,
        center_hold_sec=center_hold_seg_sec,
    )

    ile = int(sg["i_left_end"])
    i_reach = sg.get("i_reach")

    if i_reach is None or ile >= n:
        return {
            "status": "skip",
            "reason": "no_p4",
            "ile": str(ile),
            "ic": "",
            "extended": "",
            "orig_rows": str(n),
            "end_rows": "",
            "y_rc": "",
            "seg_status": str(sg.get("status", "")),
            "straight_frames": "",
        }

    dt = ov_seg._median_dt(times)
    # Right-lane center used for arrival detection (fixed from lane calibration by default).
    y_rc = float(right_lane_center)

    ys_det = ys_med if segment_smooth_window > 1 else list(ys_raw)
    ic_find = _first_sustained_near_center(
        times,
        ys_det,
        ile,
        n,
        y_rc,
        center_tol_m,
        center_hold_sec,
    )

    # If never returned to center in P4: do nothing (write-through).
    if ic_find is None:
        od = os.path.dirname(out_path)
        if od:
            os.makedirs(od, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in aligned:
                w.writerow(dict((k, r.get(k, "")) for k in fieldnames))
        return {
            "status": "skip",
            "reason": "p4_never_reached_right_center",
            "ile": str(ile),
            "ic": "",
            "end_rows": str(n),
            "orig_rows": str(n),
            "y_rc": "{:.6f}".format(y_rc),
            "seg_status": str(sg.get("status", "")),
            "straight_frames": "",
            "n_fixed": "0",
            "max_delta_m": "0",
        }

    ic_idx = int(ic_find)
    out_rows: List[Dict[str, str]] = [dict(r) for r in aligned]
    work_y = [float(ys_raw[i]) for i in range(n)]

    # Only adjust on [ic_idx, n): if right wheel presses right edge.
    n_fixed, max_delta = _apply_right_edge_fix(
        work_y,
        ic_idx,
        n,
        ego_half_width_m=ego_half_width_m,
        right_edge_y=right_edge_y,
        margin_m=0.01,
        repair_inward_m=0.02,
        merge_gap_frames=3,
        pad_frames=18,
        median_window=11,
        ma_window=13,
        span_edge_taper=15,
    )
    # Keep legacy option: extra MA smoothing (defaults to 11) after span-based repair.
    if n_fixed > 0 and int(smooth_window) > 1:
        _smooth_y_span(work_y, ic_idx, n, smooth_window)

    for i in range(ic_idx, n):
        out_rows[i]["ego_pos_y"] = "{:.6f}".format(work_y[i])

    od = os.path.dirname(out_path)
    if od:
        os.makedirs(od, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(dict((k, r.get(k, "")) for k in fieldnames))

    return {
        "status": "ok",
        "ile": str(ile),
        "ic": str(ic_idx),
        "extended": "0",
        "end_rows": str(n),
        "orig_rows": str(n),
        "y_rc": "{:.6f}".format(y_rc),
        "seg_status": str(sg.get("status", "")),
        "straight_frames": "",
        "n_fixed": str(n_fixed),
        "max_delta_m": "{:.6f}".format(max_delta),
        "reason": "",
    }


def main():
    ap = argparse.ArgumentParser(
        description="P4: fill lateral to right-lane center, trim/extend straight 2s, smooth corridor.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(_REPO, "overtaking", "selected"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO, "overtaking", "outputs", "overtaking_p4_return_fix"),
    )
    ap.add_argument("--ego_half_width_m", type=float, default=0.9)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.75)
    ap.add_argument(
        "--right_lane_center",
        type=float,
        default=-7.625,
        help="Right-lane center (ego_pos_y) used for 'returned to center' detection.",
    )
    ap.add_argument(
        "--right_edge_y",
        type=float,
        default=-9.30,
        help="Right lane outer edge y. Right wheel presses when ego_pos_y - hw <= right_edge_y.",
    )
    ap.add_argument("--y_center_p2_end", type=float, default=None)
    ap.add_argument("--p2_end_mode", choices=("fixed", "geometry"), default="fixed")
    ap.add_argument(
        "--segment_smooth_window",
        type=int,
        default=5,
        help="Moving median window for segmentation (boundary detection only).",
    )
    ap.add_argument(
        "--center_hold_seg_sec",
        type=float,
        default=0.15,
        help="Sustain-hold seconds for overtaking segmentation (passed to segment_indices).",
    )
    ap.add_argument("--edge_eps_m", type=float, default=0.08)
    ap.add_argument(
        "--center_tol_m",
        type=float,
        default=0.15,
        help="|'ego_pos_y' - y_rc| tolerance to count as 'arrived at right-lane center'.",
    )
    ap.add_argument(
        "--center_hold_sec",
        type=float,
        default=0.2,
        help="Hold near y_rc for this long (smooth y) before declaring first arrival.",
    )
    ap.add_argument(
        "--smooth_window",
        type=int,
        default=11,
        help="Moving-average window over modified corridor (odd; 1 disables).",
    )
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    y_p2 = args.y_center_p2_end
    if args.p2_end_mode == "fixed" and y_p2 is None:
        y_p2 = -4.5

    paths = ov_seg._discover_overtaking_csvs(os.path.abspath(args.data_dir))
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    base_in = os.path.abspath(args.data_dir)
    base_out = os.path.abspath(args.out_dir)
    summaries = []

    for fp in paths:
        rel = os.path.relpath(fp, base_in).replace("\\", "/")
        outp = os.path.join(base_out, rel)
        row = process_file(
            fp,
            outp,
            ego_half_width_m=args.ego_half_width_m,
            right_y_min=args.right_y_min,
            right_y_max=args.right_y_max,
            right_lane_center=args.right_lane_center,
            right_edge_y=args.right_edge_y,
            y_center_p2_end=y_p2,
            p2_end_mode=args.p2_end_mode,
            segment_smooth_window=args.segment_smooth_window,
            edge_eps_m=args.edge_eps_m,
            center_hold_seg_sec=args.center_hold_seg_sec,
            center_tol_m=args.center_tol_m,
            center_hold_sec=args.center_hold_sec,
            smooth_window=args.smooth_window,
        )
        # If this file is not calibrated/modified, still mirror original CSV into out_dir.
        if row.get("status") != "ok":
            od = os.path.dirname(outp)
            if od:
                os.makedirs(od, exist_ok=True)
            shutil.copy2(fp, outp)
        summaries.append(dict(row, rel=rel, in_path=fp, out_path=outp))

    sum_csv = os.path.join(base_out, "p4_center_modify_summary.csv")
    keys = [
        "rel",
        "status",
        "reason",
        "seg_status",
        "ile",
        "ic",
        "extended",
        "orig_rows",
        "end_rows",
        "y_rc",
        "straight_frames",
        "n_fixed",
        "max_delta_m",
        "out_path",
    ]
    with open(sum_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in summaries:
            w.writerow({k: r.get(k, "") for k in keys})

    n_ok = sum(1 for r in summaries if r.get("status") == "ok")
    print("[OK] written:", n_ok, "/", len(summaries), "→", base_out)
    print("[OK] summary:", sum_csv)


if __name__ == "__main__":
    main()
