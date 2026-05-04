# -*- coding: utf-8 -*-
"""
Modify overtaking ``driving_data.csv`` **P4 return** lateral profile (``ego_pos_y`` only):

#. Segment P4 via ``segment_overtaking_phases.segment_indices``
   → start index ``ile`` = ``i_left_end``.
#. **Right-lane geometric center**: ``y_rc = 0.5 * (right_y_min + right_y_max)``.
#. On ``[ile, n)``, find **first sustained** proximity to ``y_rc`` using smoothed ``ego_pos_y``
   (|y - y_rc| ≤ ``center_tol_m`` for ``center_hold_sec``).
#. **Before that index** ``ic``: set ``ego_pos_y = y_rc`` (segment that never settles is entirely filled).
#. From ``ic``, hold **straight** at ``y_rc`` for ``straight_sec``
   (~``straight_sec / median(dt)`` samples); output **truncate** before any extra trailing data.
#. If center is **never** reached in P4: fill ``[ ile, n )`` with ``y_rc``, **append**
   samples for ``straight_sec`` beyond the original end, still at ``y_rc``.
#. **Smooth** the modified corridor ``[ ile, end_excl )`` with moving average + replicate paddings.

Rows are sorted by ``timestamp``. Other columns unchanged except ``ego_pos_y`` where modified.

Discovery matches ``segment_overtaking_phases``: ``exp[123]_o``, exclude ``pre_familiarization`` / ``_b``.

Example::

  python3 overtaking/scripts/modify_p4_return_right_lane_center.py \\
    --data_dir overtaking/selected_p1p3_lateral \\
    --out_dir overtaking/outputs/overtaking_p4_center \\
    --straight_sec 2.0 \\
    --center_tol_m 0.15 --center_hold_sec 0.2 \\
    --smooth_window 11
"""
from __future__ import print_function

import argparse
import csv
import os
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
    y_center_p2_end: Optional[float],
    p2_end_mode: str,
    segment_smooth_window: int,
    edge_eps_m: float,
    center_hold_seg_sec: float,
    straight_sec: float,
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
    n_straight = max(1, int(round(float(straight_sec) / max(dt, 1e-9))))
    y_rc = 0.5 * (float(right_y_min) + float(right_y_max))

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

    extended = ic_find is None
    if extended:
        ic_idx = ile
        end_excl = n + n_straight
    else:
        ic_idx = int(ic_find)
        end_excl = ic_idx + n_straight

    out_rows: List[Dict[str, str]] = [dict(r) for r in aligned]
    work_y = [float(ys_raw[i]) for i in range(n)]

    if end_excl <= n:
        out_rows = out_rows[:end_excl]
        work_y = work_y[:end_excl]
    else:
        last_ts = float(times[-1])
        last_tpl = dict(aligned[-1])
        for kk in range(n, end_excl):
            nrow = dict(last_tpl)
            nrow["timestamp"] = "{:.6f}".format(last_ts + float(kk - n + 1) * dt)
            out_rows.append(nrow)
            work_y.append(y_rc)

    nn = len(out_rows)
    for i in range(ile, nn):
        work_y[i] = y_rc

    _smooth_y_span(work_y, ile, nn, smooth_window)

    for i in range(ile, nn):
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
        "extended": "1" if extended else "0",
        "end_rows": str(nn),
        "orig_rows": str(n),
        "y_rc": "{:.6f}".format(y_rc),
        "seg_status": str(sg.get("status", "")),
        "straight_frames": str(n_straight),
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
        default=os.path.join(_REPO, "overtaking", "outputs", "overtaking_p4_right_center"),
    )
    ap.add_argument("--ego_half_width_m", type=float, default=0.9)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.75)
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
        "--straight_sec",
        type=float,
        default=2.0,
        help="Hold straight lane center after first arrival.",
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
            y_center_p2_end=y_p2,
            p2_end_mode=args.p2_end_mode,
            segment_smooth_window=args.segment_smooth_window,
            edge_eps_m=args.edge_eps_m,
            center_hold_seg_sec=args.center_hold_seg_sec,
            straight_sec=args.straight_sec,
            center_tol_m=args.center_tol_m,
            center_hold_sec=args.center_hold_sec,
            smooth_window=args.smooth_window,
        )
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
