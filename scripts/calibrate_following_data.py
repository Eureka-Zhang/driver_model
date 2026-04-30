# -*- coding: utf-8 -*-
"""
Calibrate following (跟驰) driving_data.csv trajectories.

- Longitudinal: keep ego_pos_x and all lead / headway columns unchanged.
- Lateral: ego_pos_y is pulled toward the right-lane centerline:
      y_new = y_center + lateral_scale * (y_raw - y_center)
  Default lateral_scale=0.5 reduces lateral oscillation; y_center matches right lane mid (~ -7.625 m).

- Safe defaults:
      only calibrate ego_pos_y; preserve ego_speed / ego_acceleration / ego_jerk / ego_yaw / steer.
  Direct high-order differencing of quantized simulator positions can create large spikes, so
  recomputing kinematics is opt-in via --kinematics_mode recompute.
- steer: default copies original steer. Optional Ackermann-style estimate is available via
  --steer_mode bicycle and is low-pass filtered.

Does not modify: lead_*, distance_headway, time_headway, relative_speed, ttc, throttle, brake,
longitudinal_control, control_mode, gear, lead_behavior_mode, real_world_* , frame.

python3 /home/zwx/driver_model/scripts/calibrate_following_data.py \
  --data_dir /home/zwx/driver_model/data \
  --out_dir /home/zwx/driver_model/outputs/following_calibrated \
  --yaw_mode path \
  --steer_mode bicycle \
  --kinematics_mode preserve \
  --y_smooth_window 9 \
  --steer_smooth_window 21
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import re

def discover_following_csvs(data_dir):
    """Same discovery as filter_following_right_lane (no pre_familiarization, no overtaking/_o)."""
    cands = []
    for root, _, files in os.walk(data_dir):
        if "driving_data.csv" not in files:
            continue
        p = root.replace("\\", "/")
        if "pre_familiarization" in p:
            continue
        if "overtaking" in p:
            continue
        if re.search(r"/[^/]*_o(?:/|$)", p):
            continue
        if ("following" in p) or re.search(r"/[^/]*_f(?:/|$)", p):
            cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def _parse_float(s, default=None):
    if s is None:
        return default
    s = str(s).strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def _moving_average(values, window):
    """Centered moving average with edge padding; window<=1 means no smoothing."""
    if window <= 1 or not values:
        return list(values)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / float(hi - lo))
    return out


def _finite_diff_vx_vy(ts, xs, ys, n):
    """Per-sample vx, vy in m/s (world frame)."""
    vx = [0.0] * n
    vy = [0.0] * n
    if n == 1:
        return vx, vy
    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            if dt <= 0:
                dt = 1e-6
            vx[i] = (xs[1] - xs[0]) / dt
            vy[i] = (ys[1] - ys[0]) / dt
        elif i == n - 1:
            dt = ts[n - 1] - ts[n - 2]
            if dt <= 0:
                dt = 1e-6
            vx[i] = (xs[n - 1] - xs[n - 2]) / dt
            vy[i] = (ys[n - 1] - ys[n - 2]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            if dt <= 0:
                dt = 1e-6
            vx[i] = (xs[i + 1] - xs[i - 1]) / dt
            vy[i] = (ys[i + 1] - ys[i - 1]) / dt
    return vx, vy


def _unwrap_deg_deg(prev_deg, cur_deg):
    """Unwrap heading in degrees so delta is in [-180, 180]."""
    d = cur_deg - prev_deg
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return prev_deg + d


def _speed_acc_jerk(ts, speeds, n):
    acc = [0.0] * n
    jerk = [0.0] * n
    if n == 1:
        return acc, jerk
    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            if dt <= 0:
                dt = 1e-6
            acc[i] = (speeds[1] - speeds[0]) / dt
        elif i == n - 1:
            dt = ts[n - 1] - ts[n - 2]
            if dt <= 0:
                dt = 1e-6
            acc[i] = (speeds[n - 1] - speeds[n - 2]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            if dt <= 0:
                dt = 1e-6
            acc[i] = (speeds[i + 1] - speeds[i - 1]) / dt
    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            if dt <= 0:
                dt = 1e-6
            jerk[i] = (acc[1] - acc[0]) / dt
        elif i == n - 1:
            dt = ts[n - 1] - ts[n - 2]
            if dt <= 0:
                dt = 1e-6
            jerk[i] = (acc[n - 1] - acc[n - 2]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            if dt <= 0:
                dt = 1e-6
            jerk[i] = (acc[i + 1] - acc[i - 1]) / dt
    return acc, jerk


def _steer_from_bicycle(yaw_rate_rad_s, speed, wheelbase, max_steer_rad):
    """Map yaw_rate and speed to normalized steer in [-1, 1] via delta = atan(L * kappa)."""
    v = max(float(speed), 0.2)
    kappa = float(yaw_rate_rad_s) / v
    raw = math.atan(float(wheelbase) * kappa)
    lim = max(float(max_steer_rad), 1e-3)
    return max(-1.0, min(1.0, raw / lim))


def _wrap_deg(deg):
    """Wrap angle to [-180, 180)."""
    while deg >= 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def calibrate_rows(
    rows_dicts,
    y_center,
    lateral_scale,
    y_smooth_window,
    kinematics_mode,
    yaw_mode,
    wheelbase,
    max_steer_rad,
    steer_mode,
    steer_smooth_window,
):
    """
    rows_dicts: list of dicts with CSV columns.
    Returns new list of dicts (copies).
    """
    n = len(rows_dicts)
    if n == 0:
        return []

    ts = []
    xs = []
    ys_raw = []
    for r in rows_dicts:
        ts.append(_parse_float(r.get("timestamp"), 0.0))
        xs.append(_parse_float(r.get("ego_pos_x"), 0.0))
        ys_raw.append(_parse_float(r.get("ego_pos_y"), 0.0))

    ys = [y_center + lateral_scale * (y - y_center) for y in ys_raw]
    ys = _moving_average(ys, y_smooth_window)

    # Path-derived kinematics are useful only when explicitly requested. Simulator positions
    # can have small dt / quantization artifacts, and high-order differencing amplifies them.
    vx, vy = _finite_diff_vx_vy(ts, xs, ys, n)
    speeds = []
    yaws_deg = []
    for i in range(n):
        sp = math.hypot(vx[i], vy[i])
        speeds.append(sp)
        ang = math.degrees(math.atan2(vy[i], vx[i]))
        if i == 0:
            yaws_deg.append(ang)
        else:
            yaws_deg.append(_unwrap_deg_deg(yaws_deg[i - 1], ang))

    if kinematics_mode == "recompute":
        acc, jerk = _speed_acc_jerk(ts, speeds, n)
    else:
        acc = [_parse_float(r.get("ego_acceleration"), 0.0) for r in rows_dicts]
        jerk = [_parse_float(r.get("ego_jerk"), 0.0) for r in rows_dicts]

    yaw_rad = [math.radians(y) for y in yaws_deg]
    yaw_rate = [0.0] * n
    if n >= 2:
        for i in range(n):
            if i == 0:
                dt = ts[1] - ts[0]
                if dt <= 0:
                    dt = 1e-6
                yaw_rate[i] = (yaw_rad[1] - yaw_rad[0]) / dt
            elif i == n - 1:
                dt = ts[n - 1] - ts[n - 2]
                if dt <= 0:
                    dt = 1e-6
                yaw_rate[i] = (yaw_rad[n - 1] - yaw_rad[n - 2]) / dt
            else:
                dt = ts[i + 1] - ts[i - 1]
                if dt <= 0:
                    dt = 1e-6
                yaw_rate[i] = (yaw_rad[i + 1] - yaw_rad[i - 1]) / dt

    steer_est = []
    if steer_mode == "bicycle":
        for i in range(n):
            base_speed = speeds[i] if kinematics_mode == "recompute" else _parse_float(rows_dicts[i].get("ego_speed"), speeds[i])
            steer_est.append(_steer_from_bicycle(yaw_rate[i], base_speed, wheelbase, max_steer_rad))
        steer_est = _moving_average(steer_est, steer_smooth_window)

    out = []
    for i, r in enumerate(rows_dicts):
        new_r = dict(r)
        new_r["ego_pos_y"] = "{:.6f}".format(ys[i])
        if kinematics_mode == "recompute":
            new_r["ego_speed"] = "{:.6f}".format(speeds[i])
            new_r["ego_acceleration"] = "{:.6f}".format(acc[i])
            new_r["ego_jerk"] = "{:.6f}".format(jerk[i])

        if yaw_mode == "path":
            new_r["ego_yaw"] = "{:.6f}".format(_wrap_deg(yaws_deg[i]))

        if steer_mode == "copy":
            pass
        elif steer_mode == "zero":
            new_r["steer"] = "0.0"
        else:
            st = steer_est[i]
            new_r["steer"] = "{:.6f}".format(st)
        out.append(new_r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/following_calibrated",
    )
    ap.add_argument(
        "--y_center",
        type=float,
        default=-7.625,
        help="Right lane centerline ego_pos_y (m)",
    )
    ap.add_argument(
        "--lateral_scale",
        type=float,
        default=0.5,
        help="Multiply lateral offset from y_center by this factor (0<scale<=1 dampens swing)",
    )
    ap.add_argument(
        "--y_smooth_window",
        type=int,
        default=9,
        help="Centered moving average window for calibrated ego_pos_y; 1 disables smoothing",
    )
    ap.add_argument(
        "--kinematics_mode",
        type=str,
        default="preserve",
        choices=["preserve", "recompute"],
        help="preserve keeps ego_speed/acceleration/jerk; recompute differentiates calibrated path",
    )
    ap.add_argument(
        "--yaw_mode",
        type=str,
        default="copy",
        choices=["copy", "path"],
        help="copy keeps original ego_yaw; path recomputes yaw from calibrated path",
    )
    ap.add_argument("--wheelbase", type=float, default=2.7, help="For steer estimate (m)")
    ap.add_argument(
        "--max_steer_rad",
        type=float,
        default=1.22,
        help="Maps steer to [-1,1] as delta/max (rad), ~70 deg",
    )
    ap.add_argument(
        "--steer_mode",
        type=str,
        default="copy",
        choices=["bicycle", "copy", "zero"],
        help="How to set steer after calibration",
    )
    ap.add_argument(
        "--steer_smooth_window",
        type=int,
        default=11,
        help="Centered moving average window when --steer_mode bicycle",
    )
    ap.add_argument("--max_files", type=int, default=0)
    args = ap.parse_args()

    if not (0.0 < args.lateral_scale <= 2.0):
        print("[WARN] lateral_scale unusual:", args.lateral_scale)

    paths = discover_following_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        paths = paths[: args.max_files]

    os.makedirs(args.out_dir, exist_ok=True)
    done = 0
    for fp in paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        if not fieldnames:
            continue
        calibrated = calibrate_rows(
            rows,
            y_center=args.y_center,
            lateral_scale=args.lateral_scale,
            y_smooth_window=args.y_smooth_window,
            kinematics_mode=args.kinematics_mode,
            yaw_mode=args.yaw_mode,
            wheelbase=args.wheelbase,
            max_steer_rad=args.max_steer_rad,
            steer_mode=args.steer_mode,
            steer_smooth_window=args.steer_smooth_window,
        )
        out_fp = os.path.join(args.out_dir, rel)
        out_sub = os.path.dirname(out_fp)
        if out_sub and not os.path.isdir(out_sub):
            os.makedirs(out_sub)
        with open(out_fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in calibrated:
                w.writerow(row)
        done += 1

    print("[OK] calibrated files:", done, "->", args.out_dir)


if __name__ == "__main__":
    main()
