# -*- coding: utf-8 -*-
"""
Per-driver **Residual GRU** takeover generation: keep the first N seconds of
real driver data intact, then switch to closed-loop rollout using the trained
gain+residual GRU policy (``residual_gru_v3``) stacked on top of IDM.

For each driver T1..T20:
  1. Pick the calibrated session with the smallest ``distance_headway``
     variance by default (or use ``--session_selection index`` to keep the old
     ``--session_index`` behaviour). The lead vehicle trajectory of this session
     is preserved throughout the output.
  2. Rows with ``sim_time_s < takeover_time_s`` (default 20 s) are written
     as-is (original driver behaviour during startup).
  3. From the first row where ``sim_time_s >= takeover_time_s``, the residual
     GRU policy takes over longitudinal control. It keeps the real ``lead_v``
     on every frame, rolls out the ego state closed-loop:
         a_pred = clip((1+alpha) * a_IDM(v_cf, lead_v_real, gap_cf) + delta_a)
         v_next = max(0, v + a_pred * dt)
         gap_next = gap + (lead_v_real - v) * dt
  4. Output CSV preserves all columns; ``ego_pos_y`` / steer / etc. remain from the
     original. After takeover, ``ego_pos_x`` is **refreshed** each frame so that the
     planar bumper gap implied by ``(ego_pos_x, ego_pos_y)`` and ``(lead_pos_x,
     lead_pos_y)`` matches the rolled-out ``distance_headway`` (same definition as
     ``replay/experiment.py`` DataCollector: center distance minus half the sum of
     vehicle lengths). This keeps CARLA CSV replay visually consistent with the gap
     column.

Usage::

  python3 following/gru_train/generate_residual_gru_takeover.py

  python3 following/gru_train/generate_residual_gru_takeover.py \
    --calibrated_dir /home/zwx/driver_model/following/outputs/following_calibrated \
    --model_root /home/zwx/driver_model/following/outputs/residual_gru_v3 \
    --out_dir /home/zwx/driver_model/following/outputs/residual_gru_takeover_20s \
    --takeover_time_s 20.0 \
    --session_selection min_headway_variance \
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


def _half_len_sum_from_handoff_row(row, distance_headway_m):
    """
    Match DataCollector: distance_headway = center_dist - (L_ego+L_lead)/2.
    Infer (L_ego+L_lead)/2 from the last real row before takeover.
    """
    ex = _row_value(row, "ego_pos_x")
    ey = _row_value(row, "ego_pos_y")
    lx = _row_value(row, "lead_pos_x")
    ly = _row_value(row, "lead_pos_y")
    dh = _parse_float(distance_headway_m)
    if ex is None or ey is None or lx is None or ly is None or dh is None:
        return None
    center = math.hypot(ex - lx, ey - ly)
    half_sum = center - float(dh)
    if half_sum < 0.05:
        half_sum = 4.5
    return float(half_sum)


def _ego_x_from_gap_and_lead(lx, ly, ey, dh_bumper_m, half_len_sum, ego_x_ref):
    """
    Fix ego_pos_x given fixed ego_y and lead (x,y), so planar center distance equals
    dh_bumper_m + half_len_sum. Two solutions on the circle; pick closer to ego_x_ref.
    """
    D = max(0.0, float(dh_bumper_m)) + float(half_len_sum)
    dy = float(ey) - float(ly)
    r2 = D * D - dy * dy
    if r2 <= 0.0:
        return float(ego_x_ref)
    rx = math.sqrt(r2)
    xa = lx + rx
    xb = lx - rx
    return xa if abs(xa - ego_x_ref) <= abs(xb - ego_x_ref) else xb


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
    key_list = [keys] if isinstance(keys, str) else list(keys)
    out = []
    for r in rows:
        v = _row_value(r, *key_list)
        out.append(float(v) if v is not None else float(default))
    return np.asarray(out, dtype=np.float32)


def _headway_variance(csv_path):
    vals = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = _parse_float(row.get("distance_headway"))
            if v is not None and math.isfinite(v):
                vals.append(float(v))
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return sum((x - mean) ** 2 for x in vals) / len(vals)


def _select_min_headway_variance_csv(paths):
    best_path = None
    best_var = None
    for p in paths:
        var = _headway_variance(p)
        if var is None:
            continue
        if best_var is None or var < best_var or (var == best_var and p < best_path):
            best_path = p
            best_var = var
    return best_path, best_var


def _is_within_dir(path, root):
    try:
        path_abs = os.path.abspath(path)
        root_abs = os.path.abspath(root)
        return os.path.commonpath([path_abs, root_abs]) == root_abs
    except (OSError, ValueError):
        return False


def _load_selected_sessions_csv(selected_csv, calibrated_dir):
    """Read driver_id -> csv_path from select_min_headway_variance.py output."""
    out = {}
    if not selected_csv or not os.path.isfile(selected_csv):
        return out
    with open(selected_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = str(row.get("driver_id", "")).strip()
            csv_path = str(row.get("csv_path", "")).strip()
            if did and csv_path and os.path.isfile(csv_path) and _is_within_dir(csv_path, calibrated_dir):
                out[did] = csv_path
    return out


def _parse_driver_float_overrides(text):
    """Parse comma-separated driver:value overrides, e.g. ``T16:0.025,T7:0.05``."""
    out = {}
    if not text:
        return out
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit("Bad driver override {!r}; expected DRIVER:VALUE".format(item))
        driver, value = item.split(":", 1)
        driver = driver.strip()
        try:
            out[driver] = float(value.strip())
        except ValueError:
            raise SystemExit("Bad value in driver override {!r}".format(item))
    return out


def _moving_average_edge(x, window_n):
    """Centered moving average with edge padding; window_n <= 1 returns x unchanged."""
    arr = np.asarray(x, dtype=np.float64)
    w = int(window_n)
    if w <= 1 or arr.size == 0:
        return arr.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(padded, kernel, mode="valid")


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
             "Only used with --session_selection index. Default -1 = last session.",
    )
    ap.add_argument(
        "--session_selection",
        type=str,
        default="min_headway_variance",
        choices=["min_headway_variance", "index"],
        help="How to choose the calibrated source session per driver. "
             "Default chooses the smallest distance_headway variance.",
    )
    ap.add_argument(
        "--selected_sessions_csv",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_calibrated/headway_variance_selected_min.csv",
        help="Optional CSV from select_min_headway_variance.py. Used first when "
             "--session_selection min_headway_variance; falls back to computing variance.",
    )
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--drivers", type=str, default="",
                    help="Comma-separated driver IDs (empty = all found).")
    ap.add_argument(
        "--no_ego_pos_x_refresh",
        action="store_true",
        help="Disable post-takeover ego_pos_x realignment to distance_headway (old behaviour).",
    )
    ap.add_argument(
        "--gap_anchor_drivers",
        type=str,
        default="T4,T7,T16",
        help="Comma-separated driver IDs that get weak raw-gap anchoring after takeover. "
             "Use an empty string to disable. Default targets the outlier drivers T4,T7,T16.",
    )
    ap.add_argument(
        "--gap_anchor_kp",
        type=float,
        default=0.06,
        help="Acceleration correction gain for anchored drivers: a += kp * (gap_generated - gap_raw).",
    )
    ap.add_argument(
        "--gap_anchor_clip",
        type=float,
        default=1.5,
        help="Absolute clip for the gap-anchor acceleration correction in m/s^2.",
    )
    ap.add_argument(
        "--gap_anchor_kp_overrides",
        type=str,
        default="T16:0.025",
        help="Per-driver kp overrides, e.g. T16:0.025,T7:0.05. Default softens T16.",
    )
    ap.add_argument(
        "--gap_anchor_clip_overrides",
        type=str,
        default="T16:0.6",
        help="Per-driver correction clip overrides in m/s^2. Default softens T16.",
    )
    ap.add_argument(
        "--gap_anchor_smooth_window_s",
        type=float,
        default=4.0,
        help="Smooth raw gap target before anchoring. Larger values reduce oscillation.",
    )
    ap.add_argument(
        "--gap_anchor_ramp_s",
        type=float,
        default=8.0,
        help="Seconds after takeover to ramp in the anchor correction.",
    )
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
    gap_anchor_drivers = {
        x.strip() for x in str(args.gap_anchor_drivers).split(",") if x.strip()
    }
    gap_anchor_kp_overrides = _parse_driver_float_overrides(args.gap_anchor_kp_overrides)
    gap_anchor_clip_overrides = _parse_driver_float_overrides(args.gap_anchor_clip_overrides)
    selected_session_map = {}
    if args.session_selection == "min_headway_variance":
        selected_session_map = _load_selected_sessions_csv(args.selected_sessions_csv, args.calibrated_dir)
        if selected_session_map:
            print("[INFO] loaded selected min-variance sessions: {}".format(args.selected_sessions_csv))
        elif args.selected_sessions_csv and os.path.isfile(args.selected_sessions_csv):
            print(
                "[WARN] selected_sessions_csv has no usable calibrated paths; "
                "will compute min headway variance from --calibrated_dir."
            )

    for d in selected:
        paths = by_driver.get(d, [])
        if not paths:
            print("[SKIP] {} has no sessions.".format(d))
            continue

        source_selection = ""
        source_headway_var = None
        if args.session_selection == "min_headway_variance":
            mapped_path = selected_session_map.get(d)
            if mapped_path and os.path.isfile(mapped_path):
                csv_path = mapped_path
                source_selection = "selected_sessions_csv"
                source_headway_var = _headway_variance(csv_path)
            else:
                csv_path, source_headway_var = _select_min_headway_variance_csv(paths)
                source_selection = "computed_min_headway_variance"
                if csv_path is None:
                    print("[SKIP] {}: no valid distance_headway values in calibrated sessions.".format(d))
                    continue
        else:
            # Support negative index (e.g. -1 = last session)
            idx = args.session_index if args.session_index >= 0 else len(paths) + args.session_index
            if idx < 0 or idx >= len(paths):
                print("[SKIP] {} has only {} sessions (index {} out of range).".format(
                    d, len(paths), args.session_index))
                continue
            csv_path = paths[idx]
            source_selection = "index:{}".format(args.session_index)
            source_headway_var = _headway_variance(csv_path)

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
        gap_anchor_kp_used = gap_anchor_kp_overrides.get(d, float(args.gap_anchor_kp))
        gap_anchor_clip_used = gap_anchor_clip_overrides.get(d, float(args.gap_anchor_clip))
        # Anchor against a low-pass raw gap target instead of chasing frame noise.
        smooth_window_n = max(1, int(round(max(0.0, float(args.gap_anchor_smooth_window_s)) / 0.05)))
        gap_anchor_target_arr = _moving_average_edge(gap_arr, smooth_window_n)

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
        gap_anchor_enabled = d in gap_anchor_drivers
        gap_anchor_used = 0

        half_len_sum = None
        if not args.no_ego_pos_x_refresh:
            half_len_sum = _half_len_sum_from_handoff_row(
                rows[takeover_idx - 1],
                gap_arr[takeover_idx - 1],
            )
            if half_len_sum is None:
                print(
                    "[WARN] {} cannot infer vehicle half-length sum; skip ego_pos_x refresh.".format(
                        d,
                    ),
                )

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
            gap_anchor_a = 0.0
            gap_anchor_target = float(gap_anchor_target_arr[t])
            if gap_anchor_enabled and math.isfinite(gap_anchor_target):
                # Negative when generated gap is too small: decelerate ego to recover headway.
                gap_anchor_a = float(gap_anchor_kp_used) * (float(gap_cur) - gap_anchor_target)
                ramp_s = max(0.0, float(args.gap_anchor_ramp_s))
                if ramp_s > 1e-9 and t_cur is not None:
                    ramp = max(0.0, min(1.0, (float(t_cur) - float(args.takeover_time_s)) / ramp_s))
                    gap_anchor_a *= ramp
                clip_abs = abs(float(gap_anchor_clip_used))
                if clip_abs > 0.0:
                    gap_anchor_a = max(-clip_abs, min(clip_abs, gap_anchor_a))
                a_pred = float(a_pred) + gap_anchor_a
                a_pred = float(max(policy.accel_clip[0], min(policy.accel_clip[1], a_pred)))
                gap_anchor_used += 1

            # Write closed-loop state into row
            v_next = max(0.0, v_cur + a_pred * dt)
            gap_next = max(0.0, gap_cur + (lead_v - v_cur) * dt)

            rows[t]["ego_a_long"] = _fmt(a_pred)
            rows[t]["ego_acceleration"] = _fmt(a_pred)
            rows[t]["ego_v_long"] = _fmt(v_cur)
            rows[t]["ego_speed"] = _fmt(v_cur)
            dh_write = max(gap_cur, 0.0)
            rows[t]["distance_headway"] = _fmt(dh_write)

            if half_len_sum is not None:
                lx = _row_value(rows[t], "lead_pos_x")
                ly = _row_value(rows[t], "lead_pos_y")
                ey = _row_value(rows[t], "ego_pos_y")
                x_ref = _row_value(rows[t - 1], "ego_pos_x")
                if lx is not None and ly is not None and ey is not None and x_ref is not None:
                    ex_new = _ego_x_from_gap_and_lead(
                        lx, ly, ey, dh_write, half_len_sum, x_ref,
                    )
                    rows[t]["ego_pos_x"] = "{:.2f}".format(ex_new)

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
            rows[t]["gap_anchor_a"] = _fmt(gap_anchor_a)
            rows[t]["gap_anchor_target"] = _fmt(gap_anchor_target)

            # Advance state and history
            hist = policy.push_history(hist, v_next, a_pred, gap_next, lead_v)
            v_cur = v_next
            gap_cur = gap_next
            n_pred += 1

        # Ensure fieldnames include derived columns
        for col_name in ("ego_a_long", "ego_acceleration", "ttc", "time_headway",
                         "inv_ttc", "inv_time_headway", "relative_v_long",
                         "gru_alpha", "gru_delta_a", "a_idm_base",
                         "gap_anchor_a", "gap_anchor_target"):
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
            source_selection=source_selection,
            source_headway_variance="" if source_headway_var is None else "{:.6f}".format(source_headway_var),
            model_dir=model_dir,
            out_csv=out_fp,
            total_rows=n,
            takeover_row=takeover_idx,
            takeover_time_s=args.takeover_time_s,
            n_gru_pred=n_pred,
            gap_anchor_enabled=gap_anchor_enabled,
            gap_anchor_kp=gap_anchor_kp_used if gap_anchor_enabled else "",
            gap_anchor_clip=gap_anchor_clip_used if gap_anchor_enabled else "",
            gap_anchor_smooth_window_s=args.gap_anchor_smooth_window_s if gap_anchor_enabled else "",
            gap_anchor_ramp_s=args.gap_anchor_ramp_s if gap_anchor_enabled else "",
            gap_anchor_used=gap_anchor_used,
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
