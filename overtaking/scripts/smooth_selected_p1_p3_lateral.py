# -*- coding: utf-8 -*-
"""
针对 ``overtaking/selected/<T*>/<session>/driving_data.csv``；``<session>`` 若名为 ``..._o`` / ``..._b`` 可
用 ``--filter_session_suffix`` 只跑其中一类，汇总列 ``session_suffix`` 会解析该尾缀（否则为空）。

- **仅改 lateral**：只写 ``ego_pos_y``；**纵向** ``ego_pos_x`` 等列不写回。
- **仅 P1 / P3** 内可对 **轮带压线连续段**（``--merge_gap``）做平移/夹紧与局部平滑；其它帧保持原样字符串。
- **几何目标**：优先 **对称平移**；可行区间 ``[slo, shi]`` 内取 **离 0 最近** 的端点（最小横向修正），
  再向可行域 **内侧** 收 ``--repair_inward_m``（贴边后再往里一点）。平移无解时 **夹紧** 到有效中心带边界后同样内收。
- **平滑**：跨度内 **中位数 + MA**；pad 斜坡用 **smoothstep**；跨度边缘余弦混合；对已修改区间做分段 MA，
  端点补齐优先用相邻 **原始** 帧；可选用 ``--junction_blend_frames`` 在修正段外若干帧上做 smoothstep **桥接**
  （会写入这些相邻未修帧）。
- **分段护栏（动态）**：在 P1/P3 与各衔接侧相邻处，统计 **紧靠边界的连续压线帧**（与修复同一 ``chk``），
  该段不参与修正且 pad 不得伸入。**不再**固定忽略 N 帧；可用 ``--boundary_ignore_max``（默认 **0**=不封顶）限制单侧最大忽略长度。
- **分界处**：不对 ``ie`` / ``ir`` / ``ile`` 做相位混合；可选用 ``junction_blend_frames`` **仅作用于**
  「修正区间 ↔ 仍保持原始坐标的毗邻帧」接缝（例如护栏外侧、P2 中与修正尾相邻的帧），不是跨相位分界。
- **无违规**：若 P1、P3 均未产生可修复违规段（合并后跨度为空），则输出 CSV 与各帧原始字符串一致。

Example::

    python3 overtaking/scripts/smooth_selected_p1_p3_lateral.py --selected_dir overtaking/outputs/overtaking_phase_segments_all \
      --out_dir overtaking/p1p3_lateral_all --boundary_ignore_max 0 --repair_inward_m 0.03 \
      --pad_frames 30 --span_edge_taper 20 --median_window 11 --ma_window 13 \
      --post_smooth_window 25 --junction_blend_frames 24

``--boundary_ignore_frames CAP`` aliases ``--boundary_ignore_max CAP``.
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


def discover_selected_csvs(
    selected_dir: str,
    *,
    filter_session_suffix: str = "",
) -> List[str]:
    """
    Walk ``selected_dir`` for ``.../driving_data.csv``.

    If ``filter_session_suffix`` is ``\"_o\"`` or ``\"_b\"``, keep only files whose
    parent folder name **ends with** that suffix (e.g. ``..._exp3_o``, ``..._run_b``).
    """
    selected_dir = os.path.abspath(selected_dir)
    want = (filter_session_suffix or "").strip()
    out = []
    for root, _, files in os.walk(selected_dir):
        if "driving_data.csv" not in files:
            continue
        fp = os.path.join(root, "driving_data.csv")
        if want in ("_o", "_b"):
            parent = os.path.basename(root)
            if not parent.endswith(want):
                continue
        out.append(fp)
    return sorted(out)


def parse_session_ob_suffix_from_path(csv_path: str) -> str:
    """Return ``_o``, ``_b``, or empty if parent folder name has no such trailing tag."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))
    if parent.endswith("_o"):
        return "_o"
    if parent.endswith("_b"):
        return "_b"
    return ""


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


def smooth_changed_segments_ma(
    y: List[float],
    ys_raw: Sequence[float],
    changed: List[bool],
    window: int,
) -> None:
    """MA each contiguous ``changed`` run; pad with lateral raw neighbors when available (else segment ends)."""
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
        right_pad = []
        for t in range(1, half + 1):
            idx = hi + t
            if idx < n and not changed[idx]:
                right_pad.append(float(ys_raw[idx]))
            else:
                right_pad.append(seg[-1])
        padded = left_pad + seg + right_pad
        sm = moving_average(padded, w)
        for j in range(lo, hi + 1):
            y[j] = sm[half + (j - lo)]
        i = hi + 1


def junction_blend_into_raw_neighbors(
    ys_out: List[float],
    ys_raw: Sequence[float],
    changed: List[bool],
    n: int,
    blend_frames: int,
) -> None:
    """
    Blend ``blend_frames`` raw neighbors toward each repaired run edge (smoothstep).
    Indices that started unchanged get written so CSV matches ``ys_out``.
    """
    jb = max(0, int(blend_frames))
    if jb <= 0:
        return
    eligible_before = list(changed)
    runs_snap: List[Tuple[int, int, float, float]] = []
    i = 0
    while i < n:
        if not eligible_before[i]:
            i += 1
            continue
        lo = i
        while i + 1 < n and eligible_before[i + 1]:
            i += 1
        hi = i
        runs_snap.append((lo, hi, float(ys_out[lo]), float(ys_out[hi])))
        i = hi + 1
    if not runs_snap:
        return
    scale = float(jb + 1)
    for lo, hi, ref_l, ref_r in runs_snap:
        for u in range(1, jb + 1):
            idx = lo - u
            if idx >= 0 and not eligible_before[idx]:
                wx = _smoothstep(1.0 - float(u) / scale)
                ys_out[idx] = (1.0 - wx) * float(ys_raw[idx]) + wx * ref_l
                changed[idx] = True
        for u in range(1, jb + 1):
            idx = hi + u
            if idx < n and not eligible_before[idx]:
                wx = _smoothstep(1.0 - float(u) / scale)
                ys_out[idx] = (1.0 - wx) * float(ys_raw[idx]) + wx * ref_r
                changed[idx] = True


def symmetric_shift_for_wheel_band(
    y_samples: Sequence[float],
    ymin: float,
    ymax: float,
    hw: float,
    margin_m: float,
    repair_inward_m: float,
) -> Optional[float]:
    """
    Feasible additive shift on center y is ``[slo, shi]`` (per-sample wheel in band).
    Pick **minimum |shift|** (closest to 0), then move **inward** into the interval by
    ``repair_inward_m`` so wheels sit just inside the margin band, not mid-slack.
    """
    if not y_samples:
        return 0.0
    rlo = float(ymin) + float(margin_m)
    rhi = float(ymax) - float(margin_m)
    hwf = float(hw)
    slo = max(rlo - float(y) + hwf for y in y_samples)
    shi = min(rhi - float(y) - hwf for y in y_samples)
    if slo > shi + 1e-8:
        return None
    inward = max(0.0, float(repair_inward_m))
    if slo <= 0.0 <= shi:
        shift = 0.0
    elif slo > 0.0:
        shift = min(slo + inward, shi)
    else:
        shift = max(shi - inward, slo)
    return max(slo, min(shi, shift))


def clamp_centers_wheel_band(
    ys: Sequence[float],
    ymin: float,
    ymax: float,
    hw: float,
    margin_m: float,
    repair_inward_m: float,
) -> List[float]:
    """Clamp to feasible center range; when hitting a bound, nudge **inward** by ``repair_inward_m``."""
    rlo = float(ymin) + float(margin_m)
    rhi = float(ymax) - float(margin_m)
    hwf = float(hw)
    inward = max(0.0, float(repair_inward_m))
    if hwf <= 0.0:
        out = []
        for y in ys:
            fy = float(y)
            if fy < rlo:
                cy = rlo + inward
            elif fy > rhi:
                cy = rhi - inward
            else:
                cy = fy
            out.append(max(rlo, min(rhi, cy)))
        return out
    lo_center = rlo + hwf
    hi_center = rhi - hwf
    out = []
    for y in ys:
        fy = float(y)
        if fy < lo_center:
            cy = lo_center + inward
        elif fy > hi_center:
            cy = hi_center - inward
        else:
            cy = fy
        out.append(max(lo_center, min(hi_center, cy)))
    return out


def violation_flags_in_range(
    ys: Sequence[float],
    idx0: int,
    idx1: int,
    checker,
) -> List[bool]:
    n = len(ys)
    return [checker(float(ys[i])) for i in range(idx0, min(idx1, n))]


def count_consecutive_violations_prefix(
    ys_raw: Sequence[float],
    lo: int,
    hi_excl: int,
    checker,
) -> int:
    """From ``ys_raw[lo]`` forward until first non-violation or ``hi_excl``."""
    n = len(ys_raw)
    hi_excl = max(lo, min(int(hi_excl), n))
    lo = max(0, int(lo))
    c = 0
    for i in range(lo, hi_excl):
        if checker(float(ys_raw[i])):
            c += 1
        else:
            break
    return c


def count_consecutive_violations_suffix(
    ys_raw: Sequence[float],
    lo: int,
    hi_excl: int,
    checker,
) -> int:
    """From ``ys_raw[hi_excl - 1]`` backward until first non-violation or below ``lo``."""
    n = len(ys_raw)
    hi_excl = max(lo, min(int(hi_excl), n))
    lo = max(0, int(lo))
    if lo >= hi_excl:
        return 0
    c = 0
    for i in range(hi_excl - 1, lo - 1, -1):
        if checker(float(ys_raw[i])):
            c += 1
        else:
            break
    return c


def capped_ignore(k: int, cap: int) -> int:
    if cap <= 0:
        return k
    return min(k, int(cap))


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
    """Smoothstep ramps of additive shift in pads; full shift on ``[s,e]``."""
    out = []
    sh = float(shift)
    for i in range(S, E + 1):
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
                w_s = _smoothstep(u)
                y = y0 + w_s * sh
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
                u = float(i - S + 1) / float(nleft + 1)
                w = _smoothstep(u)
                y = y0 + w * d0
        elif i > e:
            nright = E - e
            if nright <= 0:
                y = float(clamped_core[-1])
            else:
                u = float(E - i + 1) / float(nright + 1)
                w_s = _smoothstep(u)
                y = y0 + w_s * d1
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
    repair_inward_m: float,
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
        shift = symmetric_shift_for_wheel_band(
            samples, ymin, ymax, hw, margin_m, repair_inward_m
        )
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
            clamped = clamp_centers_wheel_band(
                samples, ymin, ymax, hw, margin_m, repair_inward_m
            )
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
    merge_gap: int,
    pad_frames: int,
    span_edge_taper: int,
    boundary_ignore_max: int,
    repair_inward_m: float,
    post_smooth_window: int,
    junction_blend_frames: int,
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

    cap = max(0, int(boundary_ignore_max))
    ia_p1_dyn = count_consecutive_violations_suffix(ys_raw, 0, ie, chk_r)
    ia_p1 = capped_ignore(ia_p1_dyn, cap)

    n_p1 = 0
    if ie > ia_p1:
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
            boundary_ignore_after=ia_p1,
            repair_inward_m=repair_inward_m,
        )

    ot_ir = int(reach_raw) if reach_raw is not None else None
    ot_il = int(ile_raw) if ile_raw is not None else 0
    ile_h = min(ot_il, n)

    ib_p3_dyn = ia_p3_dyn = 0
    if ot_ir is not None and ile_h > ot_ir:
        ib_p3_dyn = count_consecutive_violations_prefix(ys_raw, ot_ir, ile_h, chk_l)
        ia_p3_dyn = count_consecutive_violations_suffix(ys_raw, ot_ir, ile_h, chk_l)
    ib_p3 = capped_ignore(ib_p3_dyn, cap)
    ia_p3 = capped_ignore(ia_p3_dyn, cap)

    n_p3 = 0
    if ot_ir is not None and ot_ir < n and ile_h > ot_ir + ib_p3 + ia_p3:
        n_p3 = apply_violation_spans_phase(
            ys_raw,
            changed,
            ys_out,
            ot_ir,
            ile_h,
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
            boundary_ignore_before=ib_p3,
            boundary_ignore_after=ia_p3,
            repair_inward_m=repair_inward_m,
        )

    smooth_changed_segments_ma(ys_out, ys_raw, changed, post_smooth_window)
    junction_blend_into_raw_neighbors(
        ys_out,
        ys_raw,
        changed,
        n,
        junction_blend_frames,
    )

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
        description=(
            "P1+P3: lateral repair on violation spans only; optional junction blend beside repaired runs "
            "(no ie/ir/ile phase knobs)."
        ),
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
    ap.add_argument(
        "--repair_inward_m",
        type=float,
        default=0.03,
        help=(
            "After minimal-feasible lateral fix (touching margin band), nudge this much (m) "
            "further into the feasible band; 0 = hug margin only."
        ),
    )
    ap.add_argument("--median_window", type=int, default=11)
    ap.add_argument("--ma_window", type=int, default=13)
    ap.add_argument("--merge_gap", type=int, default=3)
    ap.add_argument(
        "--pad_frames",
        type=int,
        default=18,
        help="Pads each violation span symmetrically inside the phase before smooth.",
    )
    ap.add_argument(
        "--span_edge_taper",
        type=int,
        default=15,
        help="Cosine taper (frames) blending filtered signal back toward y_adj at span ends.",
    )
    ap.add_argument(
        "--post_smooth_window",
        type=int,
        default=20,
        help=(
            "After all phase repairs, MA each contiguous repaired run (pad with lateral raw ys when adjacent). "
            "1 disables."
        ),
    )
    ap.add_argument(
        "--junction_blend_frames",
        type=int,
        default=8,
        help=(
            "After post MA: smoothstep-bridge each repaired run outward up to N raw frames toward the run edge; "
            "0 disables."
        ),
    )
    ap.add_argument(
        "--boundary_ignore_max",
        type=int,
        default=0,
        dest="boundary_ignore_max",
        metavar="CAP",
        help=(
            "Cap dynamic boundary ignore run length per side (0 = uncapped): "
            "P1 trims trailing violations touching P2; P3 trims head/tail contiguous violations touching P2/P4."
        ),
    )
    ap.add_argument(
        "--boundary_ignore_frames",
        type=int,
        default=None,
        metavar="CAP",
        help=(
            "Deprecated alias: same as ``--boundary_ignore_max`` (if passed, overrides default). "
            "Prefer ``--boundary_ignore_max``."
        ),
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="If >0, process at most this many files (after suffix filter, in sorted order).",
    )
    ap.add_argument(
        "--filter_session_suffix",
        type=str,
        default="",
        choices=("", "_o", "_b"),
        help=(
            "If set to _o or _b, only process sessions whose **parent folder name** ends with "
            "that suffix (e.g. ..._exp3_o / ..._run_b). Empty = all sessions."
        ),
    )
    args = ap.parse_args()
    bij_max = int(args.boundary_ignore_max)
    if args.boundary_ignore_frames is not None:
        bij_max = int(args.boundary_ignore_frames)

    y_p2 = args.y_center_p2_end
    if args.p2_end_mode == "fixed" and y_p2 is None:
        y_p2 = -4.5

    paths = discover_selected_csvs(
        args.selected_dir,
        filter_session_suffix=args.filter_session_suffix,
    )
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
                merge_gap=args.merge_gap,
                pad_frames=args.pad_frames,
                span_edge_taper=args.span_edge_taper,
                boundary_ignore_max=bij_max,
                repair_inward_m=args.repair_inward_m,
                post_smooth_window=args.post_smooth_window,
                junction_blend_frames=args.junction_blend_frames,
            )
        )
        summaries.append(
            dict(
                row,
                rel=rel,
                in_path=fp,
                out_path=outp,
                session_suffix=parse_session_ob_suffix_from_path(fp),
            )
        )

    sum_csv = os.path.join(base_out, "p1p3_lateral_summary.csv")
    os.makedirs(base_out, exist_ok=True)
    keys = [
        "rel",
        "session_suffix",
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
