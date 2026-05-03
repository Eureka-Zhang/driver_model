# -*- coding: utf-8 -*-
"""
针对 ``overtaking/selected/<T*>/<session>/driving_data.csv``：

- **仅改 lateral**：只写 ``ego_pos_y``；**纵向** ``ego_pos_x`` 等列不写回。
- **仅 P1 / P3** 内可对 **轮带压线连续段**（``--merge_gap``）做平移/夹紧与局部平滑；其它帧保持原样字符串。
- **平滑**：仅在违规扩展跨度内（中位数 + MA），跨度边缘余弦加权。
- **分段护栏**：相位内侧各留出 ``--boundary_ignore_frames``（默认 **5**）帧：压线不参与修正，
  pad 也不得伸入，减轻分段处横向错位。
- **P3 头 / P1 尾 / P3 尾**：用 ``--boundary_blend`` 帧做 smoothstep，
  分别靠向 ``ir-1``、``ys[ie]``、``ys[ile]``（均为原始轨迹），衔接 P2/P4。

Example::

    python3 overtaking/scripts/smooth_selected_p1_p3_lateral.py \\
      --selected_dir overtaking/selected \\
      --out_dir overtaking/selected_p1p3_lateral \\
      --boundary_ignore_frames 5 \\
      --pad_frames 12 --span_edge_taper 6 \\
      --median_window 7 --ma_window 9 --boundary_blend 10
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import segment_overtaking_phases as seg


def discover_selected_csvs(selected_dir: str) -> List[str]:
    selected_dir = os.path.abspath(selected_dir)
    out = []
    for root, _, files in os.walk(selected_dir):
        if "driving_data.csv" in files:
            out.append(os.path.join(root, "driving_data.csv"))
    return sorted(out)


def parse_float(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def forward_fill_series(rows: List[Dict[str, str]]) -> Tuple[List[float], List[float]]:
    """Returns ``(ego_pos_y_series, timestamp_series)``."""
    ys = []
    ts = []
    last_y = 0.0
    last_t = 0.0
    has_y = False
    has_t = False
    for r in rows:
        y = parse_float(r, "ego_pos_y")
        t = parse_float(r, "timestamp")
        if y is not None:
            last_y = float(y)
            has_y = True
        if t is not None:
            last_t = float(t)
            has_t = True
        ys.append(last_y if has_y else 0.0)
        ts.append(last_t if has_t else 0.0)
    return ys, ts


def median_filter(vals: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(vals)
    w = int(window) | 1
    half = w // 2
    n = len(vals)
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = sorted(vals[lo:hi])
        out.append(chunk[len(chunk) // 2])
    return out


def moving_average(vals: Sequence[float], window: int) -> List[float]:
    if window <= 1:
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


def _smoothstep(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def symmetric_shift_for_wheel_band(
    y_samples: Sequence[float],
    ymin: float,
    ymax: float,
    hw: float,
    margin_m: float,
) -> Optional[float]:
    if not y_samples:
        return 0.0
    rlo = float(ymin) + float(margin_m)
    rhi = float(ymax) - float(margin_m)
    hwf = float(hw)
    slo = max(rlo - float(y) + hwf for y in y_samples)
    shi = min(rhi - float(y) - hwf for y in y_samples)
    if slo <= shi + 1e-8:
        return 0.5 * (slo + shi)
    return None


def clamp_centers_wheel_band(
    ys: Sequence[float],
    ymin: float,
    ymax: float,
    hw: float,
    margin_m: float,
) -> List[float]:
    rlo = float(ymin) + float(margin_m)
    rhi = float(ymax) - float(margin_m)
    hwf = float(hw)
    if hwf <= 0.0:
        return [max(rlo, min(rhi, float(y))) for y in ys]
    lo_center = rlo + hwf
    hi_center = rhi - hwf
    return [max(lo_center, min(hi_center, float(y))) for y in ys]


def violation_flags_in_range(
    ys: Sequence[float],
    idx0: int,
    idx1: int,
    checker,
) -> List[bool]:
    n = len(ys)
    return [checker(float(ys[i])) for i in range(idx0, min(idx1, n))]


def contiguous_runs(mask: Sequence[bool], base_idx: int) -> List[Tuple[int, int]]:
    runs = []
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


def merge_runs_nearby(runs: List[Tuple[int, int]], merge_gap_frames: int) -> List[List[int]]:
    if not runs:
        return []
    runs = sorted(runs)
    merged = [[runs[0][0], runs[0][1]]]
    for s0, e0 in runs[1:]:
        ps, pe = merged[-1]
        if s0 - pe - 1 <= merge_gap_frames:
            merged[-1][1] = max(pe, e0)
        else:
            merged.append([s0, e0])
    return merged


def _expand_interval(s: int, e: int, pad: int, lo: int, hi: int) -> Tuple[int, int]:
    S = max(lo, s - pad)
    E = min(hi - 1, e + pad)
    return S, E


def build_adjusted_span_shift(
    ys_raw: Sequence[float],
    S: int,
    E: int,
    s: int,
    e: int,
    shift: float,
) -> List[float]:
    """Linear ramp of additive shift in pads; full shift on [s,e]."""
    out = []
    sh = float(shift)
    for i in range(S, E + 1):
        y0 = float(ys_raw[i])
        if i < s:
            nleft = s - S
            if nleft <= 0:
                y = y0 + sh
            else:
                frac = float(i - S + 1) / float(nleft + 1)
                y = y0 + frac * sh
        elif i > e:
            nright = E - e
            if nright <= 0:
                y = y0 + sh
            else:
                frac = float(E - i + 1) / float(nright + 1)
                y = y0 + frac * sh
        else:
            y = y0 + sh
        out.append(y)
    return out


def build_adjusted_span_clamp_pad(
    ys_raw: Sequence[float],
    S: int,
    E: int,
    s: int,
    e: int,
    clamped_core: Sequence[float],
) -> List[float]:
    """Core uses clamped_core; pads blend raw toward delta at core edges."""
    out = []
    d0 = float(clamped_core[0]) - float(ys_raw[s])
    d1 = float(clamped_core[-1]) - float(ys_raw[e])
    for i in range(S, E + 1):
        y0 = float(ys_raw[i])
        if i < s:
            nleft = s - S
            if nleft <= 0:
                y = float(clamped_core[0]) if i == s else y0
            else:
                frac = float(i - S + 1) / float(nleft + 1)
                y = y0 + frac * d0
        elif i > e:
            nright = E - e
            if nright <= 0:
                y = float(clamped_core[-1])
            else:
                frac = float(E - i + 1) / float(nright + 1)
                y = y0 + frac * d1
        else:
            y = float(clamped_core[i - s])
        out.append(y)
    return out


def cosine_span_weight(k: int, span_len: int, taper_frames: int) -> float:
    if taper_frames <= 0:
        return 1.0
    d = min(k, span_len - 1 - k)
    if d >= taper_frames:
        return 1.0
    u = float(d) / float(taper_frames)
    return 0.5 * (1.0 - math.cos(math.pi * u))


def apply_violation_spans_phase(
    ys_raw: Sequence[float],
    changed: List[bool],
    ys_out: List[float],
    phase_lo: int,
    phase_hi_exclusive: int,
    chk,
    ymin: float,
    ymax: float,
    hw: float,
    margin_m: float,
    merge_gap: int,
    pad_frames: int,
    median_w: int,
    ma_w: int,
    span_edge_taper: int,
    note_parts: List[str],
    phase_label: str,
    boundary_ignore_before: int,
    boundary_ignore_after: int,
) -> int:
    """Repair violations only in ``[inner_lo, inner_hi_excl)`` (trimmed from phase edges)."""
    if phase_lo >= phase_hi_exclusive:
        return 0
    inner_lo = phase_lo + int(boundary_ignore_before)
    inner_hi_excl = phase_hi_exclusive - int(boundary_ignore_after)
    if inner_lo >= inner_hi_excl:
        return 0

    vf = violation_flags_in_range(ys_raw, phase_lo, phase_hi_exclusive, chk)
    masked = []
    for k in range(len(vf)):
        i = phase_lo + k
        masked.append(bool(vf[k]) and (inner_lo <= i < inner_hi_excl))
    merged = merge_runs_nearby(contiguous_runs(masked, phase_lo), merge_gap)
    n_seg = 0
    for rr in merged:
        s0, e0 = int(rr[0]), int(rr[1])
        if s0 < inner_lo or e0 >= inner_hi_excl:
            continue
        samples = [float(ys_raw[i]) for i in range(s0, e0 + 1)]
        shift = symmetric_shift_for_wheel_band(samples, ymin, ymax, hw, margin_m)
        S, E = _expand_interval(s0, e0, int(pad_frames), phase_lo, phase_hi_exclusive)
        S = max(S, inner_lo)
        E = min(E, inner_hi_excl - 1)
        if S > E:
            continue
        span_len = E - S + 1
        if span_len <= 0:
            continue
        n_seg += 1
        if shift is None:
            clamped = clamp_centers_wheel_band(samples, ymin, ymax, hw, margin_m)
            buf = build_adjusted_span_clamp_pad(ys_raw, S, E, s0, e0, clamped)
            note_parts.append("%s_clamp" % phase_label)
        else:
            buf = build_adjusted_span_shift(ys_raw, S, E, s0, e0, shift)
        sm = median_filter(buf, median_w)
        sm = moving_average(sm, ma_w)
        for k in range(span_len):
            i = S + k
            w_mix = cosine_span_weight(k, span_len, span_edge_taper)
            ys_out[i] = (1.0 - w_mix) * buf[k] + w_mix * sm[k]
            changed[i] = True
    return n_seg


def process_file(
    in_path: str,
    out_path: str,
    *,
    ego_half_width_m: float,
    right_y_min: float,
    right_y_max: float,
    left_y_min: float,
    left_y_max: float,
    y_center_p2_end: Optional[float],
    p2_end_mode: str,
    segment_smooth_window: int,
    edge_eps_m: float,
    center_hold_sec: float,
    band_margin_m: float,
    median_window: int,
    ma_window: int,
    boundary_blend: int,
    merge_gap: int,
    pad_frames: int,
    span_edge_taper: int,
    boundary_ignore_frames: int,
) -> Dict[str, str]:
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        fieldnames = list(rdr.fieldnames or [])
        rows = list(rdr)
    n = len(rows)
    if n == 0 or "ego_pos_y" not in fieldnames:
        return {
            "status": "skip",
            "viol_p1_n": "",
            "viol_p3_n": "",
            "p1_shift_segments": "",
            "p3_shift_segments": "",
            "note": "empty_or_missing_ego_pos_y",
            "ie": "",
            "ir": "",
            "il": "",
        }

    ys_ff, ts = forward_fill_series(rows)
    ys_raw = list(ys_ff)
    ys_out = [float(y) for y in ys_ff]
    changed = [False] * n
    note_parts: List[str] = []

    hw = float(ego_half_width_m)
    if p2_end_mode == "geometry":
        y_thr = float(right_y_max) + hw
    else:
        y_thr = float(y_center_p2_end) if y_center_p2_end is not None else -4.5

    ys_seg = (
        seg._moving_median(list(ys_ff), int(segment_smooth_window))
        if segment_smooth_window > 1
        else list(ys_ff)
    )
    sg = seg.segment_indices(
        list(ts),
        ys_seg,
        right_y_min,
        right_y_max,
        y_thr,
        ego_half_width_m,
        segment_smooth_window=1,
        edge_eps_m=edge_eps_m,
        center_hold_sec=center_hold_sec,
    )
    ie = max(0, min(n, int(sg["i_follow_end"])))
    reach_raw = sg.get("i_reach")
    ile_raw = sg.get("i_left_end")

    chk_r = lambda y: seg._wheel_violation_right(y, right_y_min, right_y_max, hw)
    chk_l = lambda y: seg._wheel_violation_left(y, left_y_min, left_y_max, hw)

    nib = max(0, int(boundary_ignore_frames))

    n_p1 = 0
    if ie > nib:
        n_p1 = apply_violation_spans_phase(
            ys_raw,
            changed,
            ys_out,
            0,
            ie,
            chk_r,
            right_y_min,
            right_y_max,
            hw,
            band_margin_m,
            merge_gap,
            pad_frames,
            median_window,
            ma_window,
            span_edge_taper,
            note_parts,
            "p1",
            boundary_ignore_before=0,
            boundary_ignore_after=nib,
        )

    ot_ir = int(reach_raw) if reach_raw is not None else None
    ot_il = int(ile_raw) if ile_raw is not None else 0

    n_p3 = 0
    if ot_ir is not None and ot_il > ot_ir + 2 * nib and ot_ir < n:
        n_p3 = apply_violation_spans_phase(
            ys_raw,
            changed,
            ys_out,
            ot_ir,
            min(ot_il, n),
            chk_l,
            left_y_min,
            left_y_max,
            hw,
            band_margin_m,
            merge_gap,
            pad_frames,
            median_window,
            ma_window,
            span_edge_taper,
            note_parts,
            "p3",
            boundary_ignore_before=nib,
            boundary_ignore_after=nib,
        )

    if boundary_blend > 0 and ot_ir is not None and ot_il > ot_ir and ot_ir > 0:
        ile_h = min(ot_il, n)
        span_p3 = ile_h - ot_ir
        if span_p3 > 1:
            b = min(int(boundary_blend), span_p3 - 1)
            tgt_head = float(ys_raw[ot_ir - 1])
            for u in range(b):
                i = ot_ir + u
                t = float(u + 1) / float(b + 1)
                w = _smoothstep(t)
                ys_out[i] = (1.0 - w) * ys_out[i] + w * tgt_head
                changed[i] = True

    if boundary_blend > 0 and ie > 0 and ie < n:
        tgt = float(ys_raw[ie])
        b = min(int(boundary_blend), ie)
        for u in range(b):
            i = ie - 1 - u
            t = float(u + 1) / float(b + 1)
            w = _smoothstep(t)
            ys_out[i] = (1.0 - w) * ys_out[i] + w * tgt
            changed[i] = True

    if boundary_blend > 0 and ot_ir is not None and ot_il > ot_ir:
        ile = min(ot_il, n)
        if ile < n:
            tgt = float(ys_raw[ile])
            span = ile - ot_ir
            b = min(int(boundary_blend), span)
            for u in range(b):
                i = ile - 1 - u
                if i < ot_ir:
                    break
                t = float(u + 1) / float(b + 1)
                w = _smoothstep(t)
                ys_out[i] = (1.0 - w) * ys_out[i] + w * tgt
                changed[i] = True

    od = os.path.dirname(out_path)
    if od:
        os.makedirs(od, exist_ok=True)
    wr_kw = dict(fieldnames=fieldnames, extrasaction="ignore")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, **wr_kw)
        wr.writeheader()
        for i, r in enumerate(rows):
            rr = dict(r)
            if changed[i]:
                rr["ego_pos_y"] = "{:.6f}".format(ys_out[i])
            wr.writerow(rr)

    viol_p1 = seg._collect_violation_indices(ys_raw, (0, ie), chk_r)
    viol_p3 = []
    if ot_ir is not None and ot_il > ot_ir:
        viol_p3 = seg._collect_violation_indices(ys_raw, (ot_ir, min(ot_il, n)), chk_l)

    uniq_note = sorted(set(note_parts))

    return {
        "status": str(sg.get("status", "")),
        "viol_p1_n": str(len(viol_p1)),
        "viol_p3_n": str(len(viol_p3)),
        "p1_shift_segments": str(n_p1),
        "p3_shift_segments": str(n_p3),
        "note": ";".join(uniq_note),
        "ie": str(ie),
        "ir": "" if ot_ir is None else str(ot_ir),
        "il": str(ot_il),
    }


def main():
    ap = argparse.ArgumentParser(
        description="P1+P3: only lateral repair + smooth on wheel violation spans; smoother phase hops.",
    )
    ap.add_argument(
        "--selected_dir",
        type=str,
        default=os.path.join(_REPO, "overtaking", "selected"),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_REPO, "overtaking", "selected_p1p3_lateral"),
    )
    ap.add_argument("--ego_half_width_m", type=float, default=0.9)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.75)
    ap.add_argument("--left_y_min", type=float, default=-5.55)
    ap.add_argument("--left_y_max", type=float, default=-2.20)
    ap.add_argument("--y_center_p2_end", type=float, default=None)
    ap.add_argument("--p2_end_mode", choices=("fixed", "geometry"), default="fixed")
    ap.add_argument("--segment_smooth_window", type=int, default=5)
    ap.add_argument("--edge_eps_m", type=float, default=0.08)
    ap.add_argument("--center_hold_sec", type=float, default=0.15)
    ap.add_argument("--band_margin_m", type=float, default=0.05)
    ap.add_argument("--median_window", type=int, default=7)
    ap.add_argument("--ma_window", type=int, default=9)
    ap.add_argument(
        "--boundary_blend",
        type=int,
        default=12,
        help="P1/P3 tail frames blended toward raw y at next phase index (smoothstep).",
    )
    ap.add_argument("--merge_gap", type=int, default=3)
    ap.add_argument(
        "--pad_frames",
        type=int,
        default=14,
        help="Pads each violation span symmetrically inside the phase before smooth.",
    )
    ap.add_argument(
        "--span_edge_taper",
        type=int,
        default=8,
        help="Cosine taper (frames) blending filtered signal back toward y_adj at span ends.",
    )
    ap.add_argument(
        "--boundary_ignore_frames",
        type=int,
        default=5,
        help=(
            "Trim this many frames inside each phase boundary from repair: "
            "P1 drops last N toward P2; P3 drops first N toward P2 and last N toward P4."
        ),
    )
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    y_p2 = args.y_center_p2_end
    if args.p2_end_mode == "fixed" and y_p2 is None:
        y_p2 = -4.5

    paths = discover_selected_csvs(args.selected_dir)
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    summaries = []
    base = os.path.abspath(args.selected_dir)
    base_out = os.path.abspath(args.out_dir)
    for fp in paths:
        rel = os.path.relpath(fp, base).replace("\\", "/")
        outp = os.path.join(base_out, rel)
        row = dict(
            process_file(
                fp,
                outp,
                ego_half_width_m=args.ego_half_width_m,
                right_y_min=args.right_y_min,
                right_y_max=args.right_y_max,
                left_y_min=args.left_y_min,
                left_y_max=args.left_y_max,
                y_center_p2_end=y_p2,
                p2_end_mode=args.p2_end_mode,
                segment_smooth_window=args.segment_smooth_window,
                edge_eps_m=args.edge_eps_m,
                center_hold_sec=args.center_hold_sec,
                band_margin_m=args.band_margin_m,
                median_window=args.median_window,
                ma_window=args.ma_window,
                boundary_blend=args.boundary_blend,
                merge_gap=args.merge_gap,
                pad_frames=args.pad_frames,
                span_edge_taper=args.span_edge_taper,
                boundary_ignore_frames=args.boundary_ignore_frames,
            )
        )
        summaries.append(dict(row, rel=rel, in_path=fp, out_path=outp))

    sum_csv = os.path.join(base_out, "p1p3_lateral_summary.csv")
    os.makedirs(base_out, exist_ok=True)
    keys = [
        "rel",
        "status",
        "ie",
        "ir",
        "il",
        "viol_p1_n",
        "viol_p3_n",
        "p1_shift_segments",
        "p3_shift_segments",
        "note",
        "out_path",
    ]
    with open(sum_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in summaries:
            w.writerow({k: r.get(k, "") for k in keys})
    print("[OK]", len(summaries), "files →", base_out)


if __name__ == "__main__":
    main()
