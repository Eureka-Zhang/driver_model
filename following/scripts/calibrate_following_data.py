# -*- coding: utf-8 -*-
"""
Calibrate following (跟驰) driving_data.csv trajectories.

- Longitudinal: keep ego_pos_x and the ``distance_headway`` **CSV column** as recorded (used as
  primary gap input); clean ego_acceleration for modeling:
      median filter -> moving average -> clip to [-8, 6] m/s^2 by default.
  relative_speed is rewritten consistently as relative_v_long = lead_v_long - ego_v_long.
- Decomposed kinematics are added for longitudinal car-following modeling:
      ego_v_long, ego_v_lat, ego_a_long, ego_a_lat,
      lead_v_long, lead_v_lat, relative_v_long.
  Longitudinal/lateral velocities are projections of ego_speed / lead_speed onto the world
  road frame (fixed tangent heading; default aligns with world +x): V = speed * forward(yaw).
  The ``ego_v_*`` / ``lead_v_*`` **velocity columns are written as nonnegative** ``|v|``;
  signed projections are kept internally for ``relative_v_long`` and filtering; **accelerations
  are derivatives of the smoothed |v_long| / |v_lat| series** so ego_a_* matches the emitted
  nonnegative speed components. THW/TTC use the same magnitude conventions as before.
- Lateral: ego_pos_y is pulled toward the right-lane centerline:
      y_new = y_center + lateral_scale * (y_raw - y_center)
  Default lateral_scale=0.5 reduces lateral oscillation; y_center matches right lane mid (~ -7.625 m).

- Safe defaults:
      calibrate ego_pos_y; preserve ego_speed / ego_yaw / steer; compute speed-projected long/lat
      velocities and accelerations; set ego_acceleration to cleaned ego_a_long.
      ego_jerk is recomputed from the cleaned acceleration for later comfort analysis,
      but should not be used as a training input/output target.
  Direct high-order differencing of quantized simulator positions can create large spikes, so
  recomputing kinematics is opt-in via --kinematics_mode recompute.
- steer: default copies original steer. Optional Ackermann-style estimate is available via
  --steer_mode bicycle and is low-pass filtered.

  ``time_headway``, ``ttc`` are **rewritten** from longitudinal kinematics: gap ``dh`` prefers
  ``distance_headway`` (ego front → lead rear); else ``|ego_pos_x − lead_pos_x|``.
  ``time_headway = dh / |ego_v_long|`` uses the magnitude of the projected longitudinal speed.
  Closing TTC matches ``replay/experiment.py`` DataCollector: ``ttc = dh / (ego_speed − lead_speed)``
  when that difference is **>** 0.01 m/s (ego faster, same-lane closing). Using signed projected
  ``ego_v_long − lead_v_long`` would flip sign for typical ±180° headings vs world +x tangent.
  Columns ``inv_ttc`` and ``inv_time_headway`` are ``1/ttc`` and ``1/time_headway`` when the
  value is finite ``> 0`` and not the ``999`` sentinel; otherwise **0** (same rule as
  ``clean_following_for_imitation.py``).

Adds ``sim_time_s``: simulation elapsed time (seconds), ``row_index * sim_dt_s`` (default 0.05 s
per row, matching ``replay/experiment.py`` sync ``fixed_delta_seconds``). Kinematic differencing
(vel/accel/jerk/yaw_rate) uses this uniform timeline instead of the wall-clock ``timestamp`` column.

Does not modify: lead position/speed columns aside from derived long/lat here, throttle, brake,
longitudinal_control, control_mode, gear, lead_behavior_mode, real_world_* , frame, original ``timestamp``.

python3 /home/zwx/driver_model/following/scripts/calibrate_following_data.py \
  --data_dir /home/zwx/driver_model/data \
  --out_dir /home/zwx/driver_model/following/outputs/following_calibrated \
  --kinematics_mode preserve \
  --acc_median_window 5 \
  --acc_smooth_window 7 \
  --acc_clip_min -8 \
  --acc_clip_max 6
  
  
  cd /home/zwx/driver_model
python3 following/scripts/calibrate_following_data.py \
  --data_dir data/T9 \
  --out_dir following/outputs/following_calibrated/T9 \
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


def _row_longitudinal_gap_m(row_dict, ego_x, lead_x):
    """
    Prefer ``distance_headway`` (typically **ego front → lead rear** in the collector).

    Fallback ``|ego_x − lead_x|`` is **center-to-center** along world x when both exist;
    it is systematically larger than bumper-to-bumper headway (~vehicle half-lengths each)
    and should only be used when ``distance_headway`` is missing/invalid.
    Returns None if unusable (matches following-style / GRU invalid gap band).
    """
    dh = _parse_float(row_dict.get("distance_headway"))
    if dh is not None and math.isfinite(dh) and 1e-3 < dh < 300.0:
        return dh
    if ego_x is not None and lead_x is not None:
        try:
            d = abs(float(ego_x) - float(lead_x))
        except (TypeError, ValueError):
            d = None
        if d is not None and math.isfinite(d) and 1e-3 < d < 300.0:
            return d
    return None


# Minimum ego speed magnitude (m/s) for a numeric time_headway vs GRU sentinel 999 when ~stationary.
_THW_DENOM_EPS = 0.5


def _longitudinal_time_headway_s(dh_m, ego_v_long):
    """
    THW ≈ bumper gap / longitudinal speed magnitude: ``dh / |ego_v_long|``.
    Uses |·| because ``ego_v_long`` is projected along fixed road tangent and can oppose +x sense.
    """
    if dh_m is None or ego_v_long is None:
        return 999.0
    try:
        ev = abs(float(ego_v_long))
    except (TypeError, ValueError):
        return 999.0
    if not math.isfinite(ev) or ev < _THW_DENOM_EPS:
        return 999.0
    try:
        dh = float(dh_m)
    except (TypeError, ValueError):
        return 999.0
    if dh <= 1e-3 or not math.isfinite(dh) or dh >= 300.0:
        return 999.0
    th = dh / ev
    if not math.isfinite(th) or th <= 1e-4 or th >= 500.0:
        return 999.0
    return th


def _longitudinal_closing_ttc_s(dh_m, ego_speed_mag, lead_speed_mag):
    """
    Kinematic closing TTC, same rule as ``replay/experiment`` DataCollector.collect:
    ``relative_speed = ego_speed - lead_speed``; finite TTC only when ``relative_speed > 1e-2``.
    Signed road-frame ``ego_v_long - lead_v_long`` is **not** used here so TTC does not flip
    sign when projected longitudinal velocity is negative while scalars are nonnegative.
    """
    if dh_m is None or ego_speed_mag is None or lead_speed_mag is None:
        return 999.0
    try:
        dh = float(dh_m)
        es = float(ego_speed_mag)
        ls = float(lead_speed_mag)
    except (TypeError, ValueError):
        return 999.0
    if dh <= 1e-3 or not math.isfinite(dh) or dh >= 300.0:
        return 999.0
    dv = es - ls
    if not math.isfinite(dv) or dv <= 1e-2:
        return 999.0
    t = dh / dv
    if not math.isfinite(t) or t <= 0.0:
        return 999.0
    return min(t, 999.0)


_SENTINEL_999_TOL = 1e-3


def _reciprocal_thw_or_zero(cell):
    """
    Return 1/x for finite x>0 that is not the ~999 sentinel; else 0 (no inf from 1/999).
    Matches ``clean_following_for_imitation._reciprocal_thw_or_zero``.
    """
    x = _parse_float(cell)
    if x is None or not math.isfinite(x) or x <= 0.0:
        return 0.0
    if abs(x - 999.0) < _SENTINEL_999_TOL:
        return 0.0
    return 1.0 / x


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


def _world_speed_projected_on_road(speed_scalar, yaw_deg, road_heading_deg):
    """
    World velocity approximation V = speed * forward(yaw).
    Projects onto fixed road tangent in the horizontal plane:

        v_long = V · tangent,  v_lat = V · (−sin θ_road, cos θ_road)

    ``road_heading_deg`` is tangent direction measured CCW from +world x (deg).
    Lateral pairs with tangent (cos θ, sin θ) using left-positive normal.
    """
    sp = _parse_float(speed_scalar, 0.0)
    if sp is None or not math.isfinite(sp):
        sp = 0.0
    yaw_val = _parse_float(yaw_deg, 0.0)
    if yaw_val is None or not math.isfinite(yaw_val):
        yaw_val = 0.0

    ry = math.radians(float(road_heading_deg))
    ux, uy = math.cos(ry), math.sin(ry)

    yaw_r = math.radians(float(yaw_val))
    fx, fy = math.cos(yaw_r), math.sin(yaw_r)

    vx = sp * fx
    vy = sp * fy
    v_long = vx * ux + vy * uy
    v_lat = vx * (-uy) + vy * ux
    return float(v_long), float(v_lat)


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
    road_heading_deg,
    sim_dt_s,
):
    """
    rows_dicts: list of dicts with CSV columns.
    Returns new list of dicts (copies).

    ``sim_dt_s``: fixed simulation seconds per row; time axis for derivatives is
    ``ts[i] = i * sim_dt_s`` (wall-clock ``timestamp`` is not used for differencing).
    """
    n = len(rows_dicts)
    if n == 0:
        return []

    if sim_dt_s is None or not math.isfinite(float(sim_dt_s)) or float(sim_dt_s) <= 0:
        sim_dt_s = 0.05
    sim_dt_s = float(sim_dt_s)

    ts = [i * sim_dt_s for i in range(n)]
    xs = []
    ys_raw = []
    lead_xs = []
    lead_ys = []
    for r in rows_dicts:
        xs.append(_parse_float(r.get("ego_pos_x"), 0.0))
        ys_raw.append(_parse_float(r.get("ego_pos_y"), 0.0))
        lead_xs.append(_parse_float(r.get("lead_pos_x"), 0.0))
        lead_ys.append(_parse_float(r.get("lead_pos_y"), 0.0))

    ys = [y_center + lateral_scale * (y - y_center) for y in ys_raw]
    ys = _moving_average(ys, y_smooth_window)

    vx, vy = _finite_diff_vx_vy(ts, xs, ys, n)
    lvx, lvy = _finite_diff_vx_vy(ts, lead_xs, lead_ys, n)
    speeds = [math.hypot(vx[i], vy[i]) for i in range(n)]
    lead_speeds_fd = [math.hypot(lvx[i], lvy[i]) for i in range(n)]

    # Path kinematics yaw (used for yaw_rate / steer bicycle and optional yaw_mode path write).
    yaws_deg = []
    for i in range(n):
        ang = math.degrees(math.atan2(vy[i], vx[i]))
        if i == 0:
            yaws_deg.append(ang)
        else:
            yaws_deg.append(_unwrap_deg_deg(yaws_deg[i - 1], ang))

    # Long/lat speed: ego_speed / lead_speed projected onto fixed road tangent (see road_heading_deg).
    ego_v_long = [0.0] * n
    ego_v_lat = [0.0] * n
    lead_v_long = [0.0] * n
    lead_v_lat = [0.0] * n
    ego_scalar_ttc = [0.0] * n
    lead_scalar_ttc = [0.0] * n
    for i in range(n):
        ego_sp = (
            speeds[i]
            if kinematics_mode == "recompute"
            else _parse_float(rows_dicts[i].get("ego_speed"), speeds[i])
        )
        if ego_sp is None or not math.isfinite(ego_sp):
            ego_sp = speeds[i]
        ego_scalar_ttc[i] = float(ego_sp)

        if yaw_mode == "path":
            yaw_ego = yaws_deg[i]
        else:
            yaw_ego = _parse_float(rows_dicts[i].get("ego_yaw"))
            if yaw_ego is None or not math.isfinite(yaw_ego):
                yaw_ego = math.degrees(math.atan2(vy[i], vx[i]))

        evl, evlat = _world_speed_projected_on_road(ego_sp, yaw_ego, road_heading_deg)
        ego_v_long[i] = evl
        ego_v_lat[i] = evlat

        lead_sp = _parse_float(rows_dicts[i].get("lead_speed"), lead_speeds_fd[i])
        if lead_sp is None or not math.isfinite(lead_sp):
            lead_sp = lead_speeds_fd[i]
        lead_scalar_ttc[i] = float(lead_sp)

        yaw_lead = _parse_float(rows_dicts[i].get("lead_yaw"))
        if yaw_lead is None or not math.isfinite(yaw_lead):
            yaw_lead = math.degrees(math.atan2(lvy[i], lvx[i]))

        lvl, lvlat = _world_speed_projected_on_road(lead_sp, yaw_lead, road_heading_deg)
        lead_v_long[i] = lvl
        lead_v_lat[i] = lvlat

    # Light smoothing on projected velocities before differentiating again (accel).
    ego_v_long = _moving_average(ego_v_long, 3)
    ego_v_lat = _moving_average(ego_v_lat, 3)
    lead_v_long = _moving_average(lead_v_long, 3)
    lead_v_lat = _moving_average(lead_v_lat, 3)

    # Accel aligns with emitted ``ego_v_*`` / ``lead_v_*``: CSV stores |projected v| on both
    # axes. Differentiating signed projection while heading opposes ``road_heading_deg``
    # (e.g. ego_yaw≈180°, road_heading_deg=0) inverts longitudinal sign vs |v_long| rising.
    ego_v_long_mag = [abs(float(x)) for x in ego_v_long]
    ego_v_lat_mag = [abs(float(x)) for x in ego_v_lat]
    lead_v_long_mag = [abs(float(x)) for x in lead_v_long]
    lead_v_lat_mag = [abs(float(x)) for x in lead_v_lat]
    ego_a_long = _derivative(ts, ego_v_long_mag)
    ego_a_lat = _derivative(ts, ego_v_lat_mag)
    lead_a_long = _derivative(ts, lead_v_long_mag)
    lead_a_lat = _derivative(ts, lead_v_lat_mag)

    ego_a_long = _clean_acceleration(
        ego_a_long, acc_median_window, acc_smooth_window, acc_clip_min, acc_clip_max
    )
    ego_a_lat = _moving_average(_median_filter(ego_a_lat, acc_median_window), acc_smooth_window)
    lead_a_long = _moving_average(_median_filter(lead_a_long, acc_median_window), acc_smooth_window)
    lead_a_lat = _moving_average(_median_filter(lead_a_lat, acc_median_window), acc_smooth_window)
    relative_v_long = [lead_v_long[i] - ego_v_long[i] for i in range(n)]

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
        new_r["sim_time_s"] = "{:.6f}".format(ts[i])
        new_r["ego_pos_y"] = "{:.6f}".format(ys[i])
        new_r["ego_v_long"] = "{:.6f}".format(abs(ego_v_long[i]))
        new_r["ego_v_lat"] = "{:.6f}".format(abs(ego_v_lat[i]))
        new_r["ego_a_long"] = "{:.6f}".format(ego_a_long[i])
        new_r["ego_a_lat"] = "{:.6f}".format(ego_a_lat[i])
        new_r["lead_v_long"] = "{:.6f}".format(abs(lead_v_long[i]))
        new_r["lead_v_lat"] = "{:.6f}".format(abs(lead_v_lat[i]))
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

        dh_m = _row_longitudinal_gap_m(rows_dicts[i], xs[i], lead_xs[i])
        th_s = _longitudinal_time_headway_s(dh_m, ego_v_long[i])
        ttc_s = _longitudinal_closing_ttc_s(dh_m, ego_scalar_ttc[i], lead_scalar_ttc[i])
        new_r["time_headway"] = "{:.6f}".format(th_s)
        new_r["ttc"] = "{:.6f}".format(ttc_s)
        new_r["inv_ttc"] = "{:.9f}".format(_reciprocal_thw_or_zero(ttc_s))
        new_r["inv_time_headway"] = "{:.9f}".format(_reciprocal_thw_or_zero(th_s))
        for _drop in ("ttc_valid", "time_headway_valid"):
            new_r.pop(_drop, None)

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
        help="preserve keeps ego_speed/acceleration/jerk; recompute uses path-derived |v| when projecting",
    )
    ap.add_argument(
        "--road_heading_deg",
        type=float,
        default=0.0,
        help=(
            "World road tangent CCW from +x (degrees). Projection uses V = speed*forward(yaw). "
            "Default 0 = world +x; use 180 if your lane aligns with −x instead."
        ),
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
    ap.add_argument(
        "--sim-dt-s",
        type=float,
        default=0.05,
        metavar="SEC",
        help=(
            "Seconds of simulation time per CSV row (CARLA sync fixed_delta_seconds). "
            "Builds sim_time_s = row_index * sim_dt_s and uses this uniform axis for all "
            "kinematic derivatives (replacing wall-clock timestamp differencing)."
        ),
    )
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
            "sim_time_s",
            "ego_v_long",
            "ego_v_lat",
            "ego_a_long",
            "ego_a_lat",
            "lead_v_long",
            "lead_v_lat",
            "lead_a_long",
            "lead_a_lat",
            "relative_v_long",
            "time_headway",
            "ttc",
            "inv_ttc",
            "inv_time_headway",
        ]
        for name in extra_fields:
            if name not in fieldnames:
                fieldnames.append(name)
        # Place sim_time_s immediately after timestamp when present (readability).
        if "sim_time_s" in fieldnames and "timestamp" in fieldnames:
            fieldnames = [f for f in fieldnames if f != "sim_time_s"]
            ti = fieldnames.index("timestamp") + 1
            fieldnames.insert(ti, "sim_time_s")
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
            road_heading_deg=args.road_heading_deg,
            sim_dt_s=args.sim_dt_s,
        )
        # Ensure schema includes every key on output rows; drop stale ``*_valid`` from input lists.
        _no_valid = tuple(f for f in fieldnames if f not in ("ttc_valid", "time_headway_valid"))
        fieldnames_out = list(_no_valid)
        _seen_fn = set(fieldnames_out)
        for _row in calibrated:
            for _k in _row:
                if _k in ("ttc_valid", "time_headway_valid"):
                    continue
                if _k not in _seen_fn:
                    fieldnames_out.append(_k)
                    _seen_fn.add(_k)
        out_fp = os.path.join(args.out_dir, rel)
        out_sub = os.path.dirname(out_fp)
        if out_sub and not os.path.isdir(out_sub):
            os.makedirs(out_sub)
        with open(out_fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_out)
            w.writeheader()
            for row in calibrated:
                w.writerow(row)
        done += 1

    print("[OK] calibrated files:", done, "->", args.out_dir)


if __name__ == "__main__":
    main()
