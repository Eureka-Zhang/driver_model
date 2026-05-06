# -*- coding: utf-8 -*-
"""
Per-driver **Residual GRU** takeover generation: keep the first N seconds of
real driver data intact, then switch to closed-loop rollout using the trained
gain+residual GRU policy (``residual_gru_v3``) stacked on top of IDM.

For each driver T1..T20:
  1. Pick the **1st** calibrated session (``--session_index 0``) from
     ``following_calibrated``. The lead vehicle trajectory of this session is
     preserved throughout the output.
  2. Rows with ``sim_time_s < takeover_time_s`` (default 20 s) are written
     as-is (original driver behaviour during startup).
  3. From the first row where ``sim_time_s >= takeover_time_s``, the residual
     GRU policy takes over longitudinal control. It keeps the real ``lead_v``
     on every frame, rolls out the ego state closed-loop:
         a_pred = clip((1+alpha) * a_IDM(v_cf, lead_v_real, gap_cf) + delta_a)
         v_next = max(0, v + a_pred * dt)
         gap_next = gap + (lead_v_real - v) * dt
  4. Output CSV preserves all columns; lateral columns remain from the original.

Usage::

  python3 following/gru_train/generate_residual_gru_takeover.py

  python3 following/gru_train/generate_residual_gru_takeover.py \
    --calibrated_dir /home/zwx/driver_model/following/outputs/following_calibrated \
    --model_root /home/zwx/driver_model/following/outputs/residual_gru_v3 \
    --out_dir /home/zwx/driver_model/following/outputs/residual_gru_takeover_20s \
    --takeover_time_s 20.0 \
    --session_index 0 \
    --device cpu
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from residual_gru_policy import ResidualGRUPolicy


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


def _discover_calibrated_csvs(calibrated_dir):
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


def _load_csv_columns(rows, keys, default=0.0):
    return np.asarray([
        _row_value(r, *([k] if isinstance(k, str) else k)) if _row_value(r, *([k] if isinstance(k, str) else k)) is not None
        else float(default)
        for r in rows
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Residual GRU takeover: keep first N seconds of real data, then GRU+IDM rollout."
    )
    ap.add_argument(
        "--calibrated_dir", type=str,
        default="/home/zwx/driver_model/following/outputs/following_calibrated",
    )
    ap.add_argument(
        "--model_root", type=str,
        default="/home/zwx/driver_model/following/outputs/residual_gru_v3",
        help="Root containing <T*>/best_model.pt + model_meta.json + train_report.json.",
    )
    ap.add_argument(
        "--out_dir", type=str,
        default="/home/zwx/driver_model/following/outputs/residual_gru_takeover_20s",
    )
    ap.add_argument(
        "--takeover_time_s", type=float, default=20.0,
        help="GRU+IDM takes over at this sim_time_s (rows before are kept as-is).",
    )
    ap.add_argument(
        "--session_index", type=int, default=-1,
        help="0-based index of the session to use per driver. "
             "Default -1 = last session (sorted alphabetically).",
    )
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--drivers", type=str, default="",
                    help="Comma-separated driver IDs (empty = all found).")
    args = ap.parse_args()

    by_driver = _discover_calibrated_csvs(args.calibrated_dir)
    if not by_driver:
        raise SystemExit("No driving_data.csv found under " + args.calibrated_dir)

    if args.drivers:
        selected = [x.strip() for x in args.drivers.split(",") if x.strip()]
    else:
        selected = sorted(by_driver.keys(),
                          key=lambda x: int(x[1:]) if x[1:].isdigit() else 9999)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []

    for d in selected:
        paths = by_driver.get(d, [])
        if not paths:
            print("[SKIP] {} has no sessions.".format(d))
            continue
        # Support negative index (e.g. -1 = last session)
        idx = args.session_index if args.session_index >= 0 else len(paths) + args.session_index
        if idx < 0 or idx >= len(paths):
            print("[SKIP] {} has only {} sessions (index {} out of range).".format(
                d, len(paths), args.session_index))
            continue

        csv_path = paths[idx]
        model_dir = os.path.join(args.model_root, d)
        needed = ("best_model.pt", "model_meta.json", "train_report.json")
        if not all(os.path.isfile(os.path.join(model_dir, f)) for f in needed):
            print("[SKIP] {}: incomplete model at {}".format(d, model_dir))
            continue

        # Load per-driver policy
        try:
            policy = ResidualGRUPolicy(model_dir=model_dir, device=args.device)
        except Exception as e:
            print("[SKIP] {}: failed to load policy ({})".format(d, e))
            continue

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
        if not rows:
            print("[SKIP] {} empty CSV: {}".format(d, csv_path))
            continue

        n = len(rows)
        L = policy.seq_len

        # Find takeover row
        takeover_idx = n
        for i in range(n):
            t = _parse_float(rows[i].get("sim_time_s"))
            if t is not None and t >= args.takeover_time_s:
                takeover_idx = i
                break
        if takeover_idx >= n:
            print("[WARN] {} all rows have sim_time_s < {} — original only.".format(
                d, args.takeover_time_s))
            continue

        # Need at least L rows of past history before takeover
        if takeover_idx < L:
            print("[SKIP] {} takeover at row {} < seq_len {} (not enough history).".format(
                d, takeover_idx, L))
            continue

        # --- Build initial history window from the L rows before takeover ---
        def col(name, default=0.0):
            out = np.zeros(n, dtype=np.float32)
            for i, r in enumerate(rows):
                v = _row_value(r, name)
                out[i] = float(v) if v is not None else float(default)
            return out

        v_arr = col("ego_v_long")
        a_arr = col("ego_a_long")
        gap_arr = col("distance_headway")
        rel_v_arr = col("relative_v_long")
        lead_v_arr = col("lead_v_long")
        inv_ttc_arr = col("inv_ttc")
        inv_thw_arr = col("inv_time_headway")

        hist = policy.init_history(
            v_seq=v_arr[takeover_idx - L:takeover_idx],
            a_seq=a_arr[takeover_idx - L:takeover_idx],
            gap_seq=gap_arr[takeover_idx - L:takeover_idx],
            lead_v_seq=lead_v_arr[takeover_idx - L:takeover_idx],
            rel_v_seq=rel_v_arr[takeover_idx - L:takeover_idx],
            inv_ttc_seq=inv_ttc_arr[takeover_idx - L:takeover_idx],
            inv_thw_seq=inv_thw_arr[takeover_idx - L:takeover_idx],
        )

        # Initial state: use last real row right before takeover
        v_cur = float(v_arr[takeover_idx - 1])
        gap_cur = float(gap_arr[takeover_idx - 1])

        n_pred = 0
        for t in range(takeover_idx, n):
            # Determine dt from sim_time_s if available
            t_cur = _parse_float(rows[t].get("sim_time_s"))
            t_prev = _parse_float(rows[t - 1].get("sim_time_s"))
            dt = 0.05
            if t_cur is not None and t_prev is not None:
                dt = max(0.01, min(0.2, t_cur - t_prev))

            lead_v = float(lead_v_arr[t])  # preserved from real session

            a_pred, info = policy.step(v_cur, gap_cur, lead_v, hist)

            # Write closed-loop state into row
            v_next = max(0.0, v_cur + a_pred * dt)
            gap_next = gap_cur + (lead_v - v_cur) * dt

            rows[t]["ego_a_long"] = _fmt(a_pred)
            rows[t]["ego_acceleration"] = _fmt(a_pred)
            rows[t]["ego_v_long"] = _fmt(v_cur)
            rows[t]["ego_speed"] = _fmt(v_cur)
            rows[t]["distance_headway"] = _fmt(max(gap_cur, 0.0))

            rel_v = lead_v - v_cur
            rows[t]["relative_v_long"] = _fmt(rel_v)
            if "relative_speed" in rows[t]:
                rows[t]["relative_speed"] = _fmt(rel_v)

            # TTC / THW
            if gap_cur > 0.5 and v_cur > lead_v and (v_cur - lead_v) > 0.01:
                ttc = min(gap_cur / (v_cur - lead_v), 999.0)
            else:
                ttc = 999.0
            if v_cur > 0.1:
                thw = min(gap_cur / v_cur, 999.0)
            else:
                thw = 999.0
            rows[t]["ttc"] = _fmt(ttc)
            rows[t]["time_headway"] = _fmt(thw)
            rows[t]["inv_ttc"] = "{:.9f}".format(1.0 / ttc if ttc < 998.0 else 0.0)
            rows[t]["inv_time_headway"] = "{:.9f}".format(1.0 / thw if thw < 998.0 else 0.0)

            # Optional diagnostics
            if "alpha" not in rows[t]:
                pass
            rows[t]["gru_alpha"] = _fmt(info["alpha"])
            rows[t]["gru_delta_a"] = _fmt(info["delta_a"])
            rows[t]["a_idm_base"] = _fmt(info["a_idm"])

            # Advance state and history
            hist = policy.push_history(hist, v_next, a_pred, gap_next, lead_v)
            v_cur = v_next
            gap_cur = gap_next
            n_pred += 1

        # Ensure fieldnames include derived columns
        for col_name in ("ego_a_long", "ego_acceleration", "ttc", "time_headway",
                         "inv_ttc", "inv_time_headway", "relative_v_long",
                         "gru_alpha", "gru_delta_a", "a_idm_base"):
            if col_name not in fieldnames:
                fieldnames.append(col_name)

        out_subdir = os.path.join(args.out_dir, d)
        os.makedirs(out_subdir, exist_ok=True)
        out_fp = os.path.join(out_subdir, "driving_data.csv")
        with open(out_fp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                # Fill missing keys with empty string to keep schema uniform
                for k in fieldnames:
                    r.setdefault(k, "")
                w.writerow(r)

        print("[OK] {} -> {} (rows={}, takeover_at_row={}, gru_pred={})".format(
            d, out_fp, n, takeover_idx, n_pred))
        summary.append(dict(
            driver_id=d,
            source_csv=csv_path,
            model_dir=model_dir,
            out_csv=out_fp,
            total_rows=n,
            takeover_row=takeover_idx,
            takeover_time_s=args.takeover_time_s,
            n_gru_pred=n_pred,
        ))

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
