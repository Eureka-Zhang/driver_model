# -*- coding: utf-8 -*-
"""
Per-driver IDM takeover generation: keep the first 25 s of real driver data intact,
then switch to IDM closed-loop rollout for the remainder.

For each driver T1..T20:
  1. Pick the **3rd** calibrated session (sorted alphabetically) from following_calibrated.
  2. Rows with ``sim_time_s < 25`` are written as-is (original driver behaviour).
  3. From the first row where ``sim_time_s >= 25``, IDM takes over longitudinal control
     (using that driver's fitted parameters from ``idm_per_driver/<T*>/idm.json``).
  4. Output CSV preserves all columns; lateral columns are kept from the original.

Usage::

  python3 following/idm/generate_idm_takeover_per_driver.py

  python3 following/idm/generate_idm_takeover_per_driver.py \\
    --calibrated_dir /home/zwx/driver_model/following/outputs/following_calibrated \\
    --idm_dir /home/zwx/driver_model/following/outputs/idm_per_driver \\
    --out_dir /home/zwx/driver_model/following/outputs/idm_takeover_25s \\
    --takeover_time_s 25.0 \\
    --session_index 2
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict


def _parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _fmt(v):
    return "{:.6f}".format(float(v))


def _row_value(row, *keys):
    for k in keys:
        v = _parse_float(row.get(k))
        if v is not None:
            return v
    return None


def _idm_accel(v, vl, gap, params):
    v0 = params["v0"]
    s0 = params["s0"]
    a = params["a"]
    b = params["b"]
    T = params["T"]
    delta = params.get("delta", 4.0)
    eps = 1e-3

    s = max(gap, eps)
    dv = v - vl
    sab = math.sqrt(max(a * b, 1e-12))
    s_star = s0 + v * T + v * dv / (2.0 * sab)
    s_star = max(s_star, s0 + 1e-4)
    v0_safe = max(v0, eps)
    ratio = min(v / v0_safe, 50.0) if v0_safe > 0 else 0.0
    free_term = ratio ** delta
    acc = a * (1.0 - free_term - (s_star / s) ** 2)
    return acc


def _discover_calibrated_csvs(calibrated_dir):
    """Find all driving_data.csv grouped by driver."""
    by_driver = defaultdict(list)
    for root, _, files in os.walk(calibrated_dir):
        if "driving_data.csv" not in files:
            continue
        p = root.replace("\\", "/")
        m = re.search(r"/(T\d+)/", p)
        if not m:
            continue
        by_driver[m.group(1)].append(os.path.join(root, "driving_data.csv"))
    for d in by_driver:
        by_driver[d].sort()
    return by_driver


def main():
    ap = argparse.ArgumentParser(
        description="IDM takeover: keep first N seconds of real data, then IDM rollout."
    )
    ap.add_argument(
        "--calibrated_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_calibrated",
    )
    ap.add_argument(
        "--idm_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/idm_per_driver",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/idm_takeover_25s",
    )
    ap.add_argument(
        "--takeover_time_s",
        type=float,
        default=25.0,
        help="IDM takes over at this sim_time_s (rows before are kept as-is).",
    )
    ap.add_argument(
        "--session_index",
        type=int,
        default=2,
        help="0-based index of the session to pick per driver (default 2 = 3rd session).",
    )
    ap.add_argument(
        "--pred_accel_clip_min", type=float, default=-8.0,
    )
    ap.add_argument(
        "--pred_accel_clip_max", type=float, default=6.0,
    )
    ap.add_argument(
        "--drivers",
        type=str,
        default="",
        help="Comma-separated driver IDs (empty = all found).",
    )
    args = ap.parse_args()

    by_driver = _discover_calibrated_csvs(args.calibrated_dir)
    if not by_driver:
        raise SystemExit("No driving_data.csv found under " + args.calibrated_dir)

    if args.drivers:
        selected = [x.strip() for x in args.drivers.split(",") if x.strip()]
    else:
        selected = sorted(by_driver.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []

    for d in selected:
        paths = by_driver.get(d, [])
        if len(paths) <= args.session_index:
            print("[SKIP] {} has only {} sessions (need index {}).".format(d, len(paths), args.session_index))
            continue

        csv_path = paths[args.session_index]
        idm_path = os.path.join(args.idm_dir, d, "idm.json")
        if not os.path.isfile(idm_path):
            print("[SKIP] {} no idm.json at {}".format(d, idm_path))
            continue

        with open(idm_path, "r", encoding="utf-8") as f:
            idm_data = json.load(f)
        idm_params = idm_data["parameters"]

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
        if not rows:
            print("[SKIP] {} empty CSV: {}".format(d, csv_path))
            continue

        n = len(rows)
        takeover_idx = n  # default: no takeover if all rows < threshold

        for i in range(n):
            t = _parse_float(rows[i].get("sim_time_s"))
            if t is not None and t >= args.takeover_time_s:
                takeover_idx = i
                break

        if takeover_idx >= n:
            print("[WARN] {} all rows have sim_time_s < {} — writing original data only.".format(
                d, args.takeover_time_s))

        # --- IDM closed-loop from takeover_idx onward ---
        # Initialize state from the row just before takeover (or takeover row itself)
        init_idx = max(0, takeover_idx - 1)
        ego_v = _row_value(rows[init_idx], "ego_v_long", "ego_speed")
        gap = _row_value(rows[init_idx], "distance_headway")
        if ego_v is None:
            ego_v = 0.0
        if gap is None:
            gap = 50.0

        n_pred = 0
        for t in range(takeover_idx, n):
            # Current row time step
            t_cur = _parse_float(rows[t].get("sim_time_s"))
            if t > takeover_idx:
                t_prev = _parse_float(rows[t - 1].get("sim_time_s"))
            else:
                t_prev = _parse_float(rows[max(0, t - 1)].get("sim_time_s"))
            dt = 0.05
            if t_cur is not None and t_prev is not None and t > takeover_idx:
                dt = max(0.0, t_cur - t_prev)

            # Update state from previous step (for t > takeover_idx)
            if t > takeover_idx:
                ego_v = max(0.0, ego_v + prev_accel * dt)
                lv_prev = _row_value(rows[t - 1], "lead_v_long", "lead_speed")
                if lv_prev is None:
                    lv_prev = 0.0
                gap = gap + (lv_prev - ego_v) * dt

            # IDM prediction at current step
            lv = _row_value(rows[t], "lead_v_long", "lead_speed")
            if lv is None:
                lv = _row_value(rows[t], "lead_speed")
            if lv is None:
                prev_accel = 0.0
                continue

            pa_raw = _idm_accel(ego_v, lv, max(gap, 0.5), idm_params)
            pa = max(args.pred_accel_clip_min, min(args.pred_accel_clip_max, pa_raw))
            prev_accel = pa

            # Write IDM outputs into row
            rows[t]["ego_a_long"] = _fmt(pa)
            rows[t]["ego_acceleration"] = _fmt(pa)
            rows[t]["ego_v_long"] = _fmt(ego_v)
            rows[t]["ego_speed"] = _fmt(ego_v)
            rows[t]["distance_headway"] = _fmt(max(gap, 0.0))

            # Derived columns
            rel_v = lv - ego_v
            rows[t]["relative_v_long"] = _fmt(rel_v)
            if "relative_speed" in rows[t]:
                rows[t]["relative_speed"] = _fmt(rel_v)

            # TTC / THW
            if gap > 0.5 and ego_v > lv and (ego_v - lv) > 0.01:
                ttc = gap / (ego_v - lv)
                ttc = min(ttc, 999.0)
            else:
                ttc = 999.0
            if ego_v > 0.1:
                thw = gap / ego_v
                thw = min(thw, 999.0)
            else:
                thw = 999.0
            rows[t]["ttc"] = _fmt(ttc)
            rows[t]["time_headway"] = _fmt(thw)
            inv_ttc = 1.0 / ttc if ttc < 998.0 else 0.0
            inv_thw = 1.0 / thw if thw < 998.0 else 0.0
            rows[t]["inv_ttc"] = "{:.9f}".format(inv_ttc)
            rows[t]["inv_time_headway"] = "{:.9f}".format(inv_thw)

            n_pred += 1

        # Ensure fieldnames include derived columns
        for col in ("ego_a_long", "ego_acceleration", "ttc", "time_headway",
                    "inv_ttc", "inv_time_headway", "relative_v_long"):
            if col not in fieldnames:
                fieldnames.append(col)

        # Write output
        out_subdir = os.path.join(args.out_dir, d)
        os.makedirs(out_subdir, exist_ok=True)
        out_fp = os.path.join(out_subdir, "driving_data.csv")
        with open(out_fp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print("[OK] {} -> {} (rows={}, takeover_at_row={}, idm_pred={})".format(
            d, out_fp, n, takeover_idx, n_pred))
        summary.append(dict(
            driver_id=d,
            source_csv=csv_path,
            out_csv=out_fp,
            total_rows=n,
            takeover_row=takeover_idx,
            takeover_time_s=args.takeover_time_s,
            n_idm_pred=n_pred,
        ))

    # Summary CSV
    if summary:
        sum_path = os.path.join(args.out_dir, "generation_summary.csv")
        with open(sum_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
        print("\n[DONE] {} drivers processed. Summary: {}".format(len(summary), sum_path))
    else:
        print("[WARN] No drivers processed.")


if __name__ == "__main__":
    main()
