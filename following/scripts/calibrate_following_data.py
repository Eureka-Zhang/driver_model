# -*- coding: utf-8 -*-
"""
Calibrate following (跟驰) driving_data.csv trajectories.

- Longitudinal: keep ego_pos_x and all lead / headway columns unchanged, but clean
  ego_acceleration for modeling:
      median filter -> moving average -> clip to [-8, 6] m/s^2 by default.
  relative_speed is rewritten consistently as relative_v_long = lead_v_long - ego_v_long.
- Decomposed kinematics are added for longitudinal car-following modeling:
      ego_v_long, ego_v_lat, ego_a_long, ego_a_lat,
      lead_v_long, lead_v_lat, relative_v_long.
  For this straight-road dataset, long/lat are approximated by x/y derivatives.
- Lateral: ego_pos_y is pulled toward the right-lane centerline:
      y_new = y_center + lateral_scale * (y_raw - y_center)
  Default lateral_scale=0.5 reduces lateral oscillation; y_center matches right lane mid (~ -7.625 m).

- Safe defaults:
      calibrate ego_pos_y; preserve ego_speed / ego_yaw / steer; compute decomposed x/y
      velocities and accelerations; set ego_acceleration to cleaned ego_a_long.
      ego_jerk is recomputed from the cleaned acceleration for later comfort analysis,
      but should not be used as a training input/output target.
  Direct high-order differencing of quantized simulator positions can create large spikes, so
  recomputing kinematics is opt-in via --kinematics_mode recompute.
- steer: default copies original steer. Optional Ackermann-style estimate is available via
  --steer_mode bicycle and is low-pass filtered.

Does not modify: lead_*, distance_headway, time_headway, ttc, throttle, brake,
longitudinal_control, control_mode, gear, lead_behavior_mode, real_world_* , frame.

python3 /home/zwx/driver_model/scripts/calibrate_following_data.py \
  --data_dir /home/zwx/driver_model/data \
  --out_dir /home/zwx/driver_model/outputs/following_calibrated \
  --kinematics_mode preserve \
  --acc_median_window 5 \
  --acc_smooth_window 7 \
  --acc_clip_min -8 \
  --acc_clip_max 6
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


def _median_filter(values, window):
    """Centered median filter with edge truncation; window<=1 means no filtering."""
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
        vals = sorted(values[lo:hi])
        out.append(vals[len(vals) // 2])
    return out


def _clip(values, lo, hi):
    return [max(lo, min(hi, v)) for v in values]


def _derivative(ts, values):
    """Central finite difference of a scalar signal."""
    n = len(values)
    out = [0.0] * n
    if n <= 1:
        return out
    for i in range(n):
        if i == 0:
            dt = ts[1] - ts[0]
            if dt <= 0:
                dt = 1e-6
            out[i] = (values[1] - values[0]) / dt
        elif i == n - 1:
            dt = ts[n - 1] - ts[n - 2]
            if dt <= 0:
                dt = 1e-6
            out[i] = (values[n - 1] - values[n - 2]) / dt
        else:
            dt = ts[i + 1] - ts[i - 1]
            if dt <= 0:
                dt = 1e-6
            out[i] = (values[i + 1] - values[i - 1]) / dt
    return out


def _clean_acceleration(values, median_window, smooth_window, clip_min, clip_max):
    values = _median_filter(values, median_window)
    values = _moving_average(values, smooth_window)
    values = _clip(values, clip_min, clip_max)
    return values


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
    acc_median_window,
    acc_smooth_window,
    acc_clip_min,
    acc_clip_max,
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
    lead_xs = []
    lead_ys = []
    for r in rows_dicts:
        ts.append(_parse_float(r.get("timestamp"), 0.0))
        xs.append(_parse_float(r.get("ego_pos_x"), 0.0))
        ys_raw.append(_parse_float(r.get("ego_pos_y"), 0.0))
        lead_xs.append(_parse_float(r.get("lead_pos_x"), 0.0))
        lead_ys.append(_parse_float(r.get("lead_pos_y"), 0.0))

    ys = [y_center + lateral_scale * (y - y_center) for y in ys_raw]
    ys = _moving_average(ys, y_smooth_window)

    # Straight-road decomposition: x is longitudinal, y is lateral.
    # Smooth positions lightly before differentiation to avoid amplifying quantization noise.
    xs_for_diff = _moving_average(xs, 3)
    ys_for_diff = _moving_average(ys, 3)
    lead_xs_for_diff = _moving_average(lead_xs, 3)
    lead_ys_for_diff = _moving_average(lead_ys, 3)
    ego_v_long = _derivative(ts, xs_for_diff)
    ego_v_lat = _derivative(ts, ys_for_diff)
    lead_v_long = _derivative(ts, lead_xs_for_diff)
    lead_v_lat = _derivative(ts, lead_ys_for_diff)
    ego_a_long = _derivative(ts, ego_v_long)
    ego_a_lat = _derivative(ts, ego_v_lat)
    lead_a_long = _derivative(ts, lead_v_long)
    lead_a_lat = _derivative(ts, lead_v_lat)

    ego_a_long = _clean_acceleration(
        ego_a_long, acc_median_window, acc_smooth_window, acc_clip_min, acc_clip_max
    )
    ego_a_lat = _moving_average(_median_filter(ego_a_lat, acc_median_window), acc_smooth_window)
    lead_a_long = _moving_average(_median_filter(lead_a_long, acc_median_window), acc_smooth_window)
    lead_a_lat = _moving_average(_median_filter(lead_a_lat, acc_median_window), acc_smooth_window)
    relative_v_long = [lead_v_long[i] - ego_v_long[i] for i in range(n)]

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

    # Use decomposed longitudinal acceleration for modeling. The original
    # ego_acceleration may be a scalar/magnitude depending on the logger, so we
    # rewrite it to the cleaned x-direction acceleration for straight-road following.
    acc = ego_a_long
    jerk = _derivative(ts, acc)

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
        new_r["ego_v_long"] = "{:.6f}".format(ego_v_long[i])
        new_r["ego_v_lat"] = "{:.6f}".format(ego_v_lat[i])
        new_r["ego_a_long"] = "{:.6f}".format(ego_a_long[i])
        new_r["ego_a_lat"] = "{:.6f}".format(ego_a_lat[i])
        new_r["lead_v_long"] = "{:.6f}".format(lead_v_long[i])
        new_r["lead_v_lat"] = "{:.6f}".format(lead_v_lat[i])
        new_r["lead_a_long"] = "{:.6f}".format(lead_a_long[i])
        new_r["lead_a_lat"] = "{:.6f}".format(lead_a_lat[i])
        new_r["relative_v_long"] = "{:.6f}".format(relative_v_long[i])
        new_r["ego_acceleration"] = "{:.6f}".format(acc[i])
        new_r["ego_jerk"] = "{:.6f}".format(jerk[i])
        if kinematics_mode == "recompute":
            new_r["ego_speed"] = "{:.6f}".format(speeds[i])

        if "relative_speed" in new_r:
            new_r["relative_speed"] = "{:.6f}".format(relative_v_long[i])

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
        "--acc_median_window",
        type=int,
        default=5,
        help="Centered median filter window for ego_acceleration; 1 disables median filtering",
    )
    ap.add_argument(
        "--acc_smooth_window",
        type=int,
        default=7,
        help="Centered moving average window for ego_acceleration; 1 disables smoothing",
    )
    ap.add_argument(
        "--acc_clip_min",
        type=float,
        default=-8.0,
        help="Minimum ego_acceleration after cleaning (m/s^2)",
    )
    ap.add_argument(
        "--acc_clip_max",
        type=float,
        default=6.0,
        help="Maximum ego_acceleration after cleaning (m/s^2)",
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
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not fieldnames:
            continue
        extra_fields = [
            "ego_v_long",
            "ego_v_lat",
            "ego_a_long",
            "ego_a_lat",
            "lead_v_long",
            "lead_v_lat",
            "lead_a_long",
            "lead_a_lat",
            "relative_v_long",
        ]
        for name in extra_fields:
            if name not in fieldnames:
                fieldnames.append(name)
        calibrated = calibrate_rows(
            rows,
            y_center=args.y_center,
            lateral_scale=args.lateral_scale,
            y_smooth_window=args.y_smooth_window,
            acc_median_window=args.acc_median_window,
            acc_smooth_window=args.acc_smooth_window,
            acc_clip_min=args.acc_clip_min,
            acc_clip_max=args.acc_clip_max,
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
