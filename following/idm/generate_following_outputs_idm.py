# -*- coding: utf-8 -*-
"""
Closed-loop following trajectory generation with **per-driver IDM** (longitudinal) and the same
lateral behaviour as the historical BC-GRU generator (jitter / pooled mean / keep CSV).

This module is **self-contained**: it does not import ``gru_train/generate_no_driver_following_outputs.py``
or ``bc_gru_features`` (logic is inlined so IDM generation does not depend on GRU training code).

- **Longitudinal after warmup**: Treiber IDM from ``<idm_dir>/<follower_driver>/idm.json``.
- **Warmup**: first ``--warmup_frames`` rows use lead longitudinal acceleration; ``ego_v_roll`` advanced
  consistently before IDM steps.
- **Lateral**: ``original_jitter`` / ``pooled_mean_smooth`` / else keep original columns.

Example::

  python3 following/idm/generate_following_outputs_idm.py \\
    --data_dir /path/to/common_case_session \\
    --idm_dir /path/to/outputs/idm_per_driver \\
    --follower_driver T5 \\
    --out_dir /path/to/out/T5 \\
    --lateral_mode original_jitter \\
    --lane_center_y -7.625 \\
    --seed 42
    
    
COMMON_CASE="/home/zwx/driver_model/following/outputs/following_calibrated/T12/行车/20260421_120610_198_exp1_f"
IDM_ROOT="/home/zwx/driver_model/following/outputs/idm_per_driver"
OUT_ROOT="/home/zwx/driver_model/following/outputs/no_driver_follow_idm_common_lead"
for i in $(seq 1 20); do
  D="T${i}"
  if [ ! -f "${IDM_ROOT}/${D}/idm.json" ]; then
    echo "skip ${D} (no idm.json)"
    continue
  fi
  python3 /home/zwx/driver_model/following/idm/generate_following_outputs_idm.py \
    --data_dir "${COMMON_CASE}" \
    --idm_dir "${IDM_ROOT}" \
    --follower_driver "${D}" \
    --out_dir "${OUT_ROOT}/${D}" \
    --lateral_mode original_jitter \
    --lane_center_y -7.625 \
    --seed 42
done
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import re
import sys

import numpy as np

# --- Row parsing / aliases (aligned with ``bc_gru_features``; inlined) ----------------------

_SENTINEL_999_TOL = 1e-3


def reciprocal_inv_feature(x):
    """1/x when finite x>0 and not ~999; else 0 (matches calibrated IL)."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    if abs(v - 999.0) < _SENTINEL_999_TOL:
        return 0.0
    return 1.0 / v


def _parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_value(row, key):
    if key == "relative_speed":
        lv = _parse_float(row.get("lead_speed"))
        ev = _parse_float(row.get("ego_speed"))
        if lv is None or ev is None:
            return None
        return lv - ev

    if key == "inv_ttc":
        vu = _parse_float(row.get("inv_ttc"))
        if vu is not None:
            return float(vu)
        tt = _parse_float(row.get("ttc"))
        if tt is not None:
            return reciprocal_inv_feature(tt)
        return None

    if key == "inv_time_headway":
        vu = _parse_float(row.get("inv_time_headway"))
        if vu is not None:
            return float(vu)
        th = _parse_float(row.get("time_headway"))
        if th is not None:
            return reciprocal_inv_feature(th)
        return None

    if key == "ttc_inverse":
        return _row_value(row, "inv_ttc")
    if key == "time_headway_inverse":
        return _row_value(row, "inv_time_headway")

    if key in row:
        return _parse_float(row.get(key))
    return None


def hydrate_following_row_aliases(row):
    """Fill canonical longitudinal keys from CARLA/export aliases (same rules as BC-GRU hydrate)."""
    r = dict(row)

    def _pf(key):
        if key not in r:
            return None
        return _parse_float(r.get(key))

    def _need(key):
        return _pf(key) is None

    if _need("ego_v_long"):
        for s in ("ego_speed", "ego_v"):
            if _pf(s) is not None:
                r["ego_v_long"] = str(r[s]).strip()
                break

    if _need("ego_a_long"):
        for s in ("ego_acceleration", "ego_accel", "accel_long"):
            if _pf(s) is not None:
                r["ego_a_long"] = str(r[s]).strip()
                break

    if _need("lead_v_long"):
        for s in ("lead_speed",):
            if _pf(s) is not None:
                r["lead_v_long"] = str(r[s]).strip()
                break

    if _need("relative_v_long"):
        if _pf("relative_speed") is not None:
            r["relative_v_long"] = str(r["relative_speed"]).strip()
        elif _pf("lead_v_long") is not None and _pf("ego_v_long") is not None:
            r["relative_v_long"] = "{:.6f}".format(
                float(_pf("lead_v_long")) - float(_pf("ego_v_long"))
            )

    if _need("time_headway"):
        dh = _pf("distance_headway")
        ev = _pf("ego_v_long")
        if dh is not None and ev is not None:
            if float(ev) < 0.5:
                r["time_headway"] = "999.0"
            else:
                r["time_headway"] = "{:.6f}".format(float(dh) / float(ev))

    if _need("ttc"):
        dh = _pf("distance_headway")
        ego = _pf("ego_v_long")
        lv = _pf("lead_v_long")
        tset = 999.0
        if dh is not None and ego is not None and lv is not None:
            try:
                dhd = float(dh)
                dv = float(ego) - float(lv)
                if dhd > 1e-3 and dv > 1e-2:
                    tc = dhd / dv
                    if math.isfinite(tc) and tc > 0.0:
                        tset = min(tc, 999.0)
            except (TypeError, ValueError):
                tset = 999.0
        r["ttc"] = "{:.6f}".format(tset)

    if _need("inv_ttc"):
        tt = _pf("ttc")
        if tt is not None:
            r["inv_ttc"] = "{:.9f}".format(reciprocal_inv_feature(tt))

    if _need("inv_time_headway"):
        th = _pf("time_headway")
        if th is not None:
            r["inv_time_headway"] = "{:.9f}".format(reciprocal_inv_feature(th))

    return r


# --- Scenario discovery & lateral pool (same filters as historical GRU generator) -------------


def _scenario_tree_path_allowed(p_sub):
    p = p_sub.replace("\\", "/")
    if "pre_familiarization" in p:
        return False
    if "overtaking" in p:
        return False
    if re.search(r"/[^/]*_o(?:/|$)", p):
        return False
    if "_b" in p:
        return False
    return ("following" in p) or bool(re.search(r"/[^/]*_f(?:/|$)", p))


def discover_following_csvs(data_dir):
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
        if "_b" in p:
            continue
        if ("following" in p) or re.search(r"/[^/]*_f(?:/|$)", p):
            cands.append(os.path.join(root, "driving_data.csv"))
    return sorted(cands)


def discover_scenario_csvs(data_dir):
    ddrv = discover_following_csvs(data_dir)
    if ddrv:
        return ddrv
    segs = []
    for root, _, files in os.walk(data_dir):
        if not _scenario_tree_path_allowed(root.replace("\\", "/")):
            continue
        for fn in files:
            if re.match(r"segment_\d+\.csv$", fn):
                segs.append(os.path.join(root, fn))
    return sorted(segs)


def discover_segment_csvs(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def discover_lateral_pool_csvs(data_dir):
    segs = discover_segment_csvs(data_dir)
    drives = discover_following_csvs(data_dir)
    return sorted(set(segs) | set(drives))


def _extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    if m:
        return m.group(1)
    return "UNKNOWN"


def _build_driver_lateral_pool(paths, lane_center):
    pool = {}
    for fp in paths:
        driver = _extract_driver_id(fp)
        vals_y = []
        vals_yaw = []
        vals_steer = []
        with open(fp, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                y = _parse_float(r.get("ego_pos_y"))
                yaw = _parse_float(r.get("ego_yaw"))
                steer = _parse_float(r.get("steer"))
                if y is not None:
                    vals_y.append(y - lane_center)
                if yaw is not None:
                    vals_yaw.append(yaw)
                if steer is not None:
                    vals_steer.append(steer)
        if not vals_y:
            continue
        yaw_base = float(np.median(vals_yaw)) if vals_yaw else 0.0
        steer_base = float(np.median(vals_steer)) if vals_steer else 0.0
        yaw_res = [v - yaw_base for v in vals_yaw] if vals_yaw else [0.0]
        steer_res = [v - steer_base for v in vals_steer] if vals_steer else [0.0]
        entry = pool.setdefault(
            driver, {"y_res": [], "yaw_res": [], "steer_res": [], "source_files": 0}
        )
        entry["y_res"].extend(vals_y)
        entry["yaw_res"].extend(yaw_res)
        entry["steer_res"].extend(steer_res)
        entry["source_files"] += 1
    return pool


def _sample_with_wrap(values, n, rng):
    if not values:
        return [0.0] * n
    if len(values) == 1:
        return [values[0]] * n
    start = rng.randint(0, len(values) - 1)
    out = []
    for i in range(n):
        out.append(values[(start + i) % len(values)])
    return out


def _moving_average(values, window):
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


def _parse_driver_id_list(s):
    if not (s and str(s).strip()):
        return []
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x and x not in out:
            out.append(x)
    return out


def _concat_pooled_residuals(lateral_pool, driver_ids):
    def _key(d):
        d = d.strip()
        if d.startswith("T") and d[1:].isdigit():
            return int(d[1:])
        return 9999

    y_all, yaw_all, steer_all = [], [], []
    for d in sorted(driver_ids, key=_key):
        ent = lateral_pool.get(d, {})
        y_all.extend(ent.get("y_res") or [])
        yaw_all.extend(ent.get("yaw_res") or [])
        steer_all.extend(ent.get("steer_res") or [])
    return y_all, yaw_all, steer_all


def _residuals_to_length(smooth_y, smooth_yaw, smooth_steer, n):
    if n <= 0:
        return [], [], []
    if not smooth_y:
        return [0.0] * n, [0.0] * n, [0.0] * n
    Ly, Lya, Lst = len(smooth_y), len(smooth_yaw), len(smooth_steer)
    if Lya == 0:
        Lya = Ly
    if Lst == 0:
        Lst = Ly
    y_out = [smooth_y[i % Ly] for i in range(n)]
    yaw_out = [smooth_yaw[i % Lya] if smooth_yaw else 0.0 for i in range(n)]
    steer_out = [smooth_steer[i % Lst] if smooth_steer else 0.0 for i in range(n)]
    return y_out, yaw_out, steer_out


def _lead_longitudinal_accel(row):
    v = _parse_float(row.get("lead_a_long"))
    if v is not None:
        return v
    v = _parse_float(row.get("lead_acceleration"))
    return v if v is not None else 0.0


def _original_ego_long_accel(row):
    v = _parse_float(row.get("ego_a_long"))
    if v is not None:
        return v
    v = _parse_float(row.get("ego_acceleration"))
    return v if v is not None else 0.0


def _bounded_posterior_speed(v_euler_end, vmin_mps):
    x = max(0.0, float(v_euler_end))
    if vmin_mps is not None and float(vmin_mps) > 0.0:
        x = max(x, float(vmin_mps))
    return x


def _clip_prediction_accel(pa_raw, lo, hi):
    a = float(lo)
    b = float(hi)
    if a > b:
        a, b = b, a
    return max(a, min(b, float(pa_raw)))


def _finalize_longitudinal_from_roll(rows, ego_v_roll, distance_offset, vmin_mps):
    if not rows:
        return
    if len(ego_v_roll) != len(rows):
        raise RuntimeError(
            "ego_v_roll length {} != rows {}".format(len(ego_v_roll), len(rows))
        )
    n = len(rows)
    ts = [_parse_float(r.get("timestamp")) for r in rows]
    acc = [_original_ego_long_accel(r) for r in rows]
    x0 = _parse_float(rows[0].get("ego_pos_x"))
    if x0 is None:
        x0 = 0.0
    x_list = [x0]
    v_eff = [_bounded_posterior_speed(float(ego_v_roll[i]), vmin_mps) for i in range(n)]
    for i in range(1, n):
        dt = 0.0
        if ts[i] is not None and ts[i - 1] is not None:
            dt = max(0.0, ts[i] - ts[i - 1])
        vp = float(v_eff[i - 1])
        vn = float(v_eff[i])
        ds = 0.5 * (vp + vn) * dt
        x_list.append(x_list[-1] - ds)
    if len(x_list) != n:
        raise RuntimeError("internal longitudinal x length mismatch")
    for i in range(n):
        sx = "{:.6f}".format(x_list[i])
        vego = float(v_eff[i])
        sv = "{:.6f}".format(vego)
        rows[i]["ego_pos_x"] = sx
        if "ego_v_long" in rows[i]:
            rows[i]["ego_v_long"] = sv
        rows[i]["ego_speed"] = sv

        lv_rs = _parse_float(rows[i].get("lead_v_long"))
        if lv_rs is None:
            lv_rs = _parse_float(rows[i].get("lead_speed"))
        if lv_rs is not None:
            rs = float(lv_rs) - vego
            if "relative_speed" in rows[i]:
                rows[i]["relative_speed"] = "{:.6f}".format(rs)
            if "relative_v_long" in rows[i]:
                rows[i]["relative_v_long"] = "{:.6f}".format(rs)

        if distance_offset is not None:
            lx = _parse_float(rows[i].get("lead_pos_x"))
            if lx is not None:
                rows[i]["distance_headway"] = "{:.6f}".format(
                    abs(x_list[i] - lx) + distance_offset
                )

        gf_i = _parse_float(rows[i].get("distance_headway"))
        ev_i = max(0.0, float(vego))
        lv_tt = lv_rs if lv_rs is not None else _parse_float(rows[i].get("lead_speed"))
        if gf_i is not None and lv_tt is not None and math.isfinite(gf_i) and math.isfinite(
            lv_tt
        ):
            _apply_tt_th_inv_to_row(rows[i], gf_i, ev_i, max(0.0, float(lv_tt)))

        if "ego_jerk" in rows[i]:
            if i == 0:
                rows[i]["ego_jerk"] = "{:.6f}".format(0.0)
            elif ts[i] is not None and ts[i - 1] is not None:
                dt = max(1e-9, ts[i] - ts[i - 1])
                j = (acc[i] - acc[i - 1]) / dt
                rows[i]["ego_jerk"] = "{:.6f}".format(j)


def _smooth_lateral_if_needed(y_res, yaw_res, steer_res, limit_abs, smooth_window):
    if not y_res:
        return y_res, yaw_res, steer_res, "driver_residual_pool"
    if max(abs(v) for v in y_res) <= limit_abs:
        return y_res, yaw_res, steer_res, "driver_residual_pool"

    y_sm = _moving_average(y_res, smooth_window)
    yaw_sm = _moving_average(yaw_res, smooth_window)
    steer_sm = _moving_average(steer_res, smooth_window)
    y_sm = [max(-limit_abs, min(limit_abs, v)) for v in y_sm]
    return y_sm, yaw_sm, steer_sm, "driver_residual_pool_smoothed_limited"


def _fmt_feat_scalar(x):
    return "{:.6f}".format(float(x))


def _fmt_inv_scalar(x):
    return "{:.9f}".format(float(x))


_THW_DENOM_EPS = 0.5
_DH_GAP_MIN = 1e-3
_DH_GAP_MAX = 300.0


def kinematic_closing_ttc(distance_m, ego_speed_mag, lead_speed_mag):
    if distance_m is None or ego_speed_mag is None or lead_speed_mag is None:
        return 999.0
    try:
        dh = float(distance_m)
        es = float(ego_speed_mag)
        ls = float(lead_speed_mag)
    except (TypeError, ValueError):
        return 999.0
    if dh <= _DH_GAP_MIN or not math.isfinite(dh) or dh >= _DH_GAP_MAX:
        return 999.0
    dv = es - ls
    if not math.isfinite(dv) or dv <= 1e-2:
        return 999.0
    t = dh / dv
    if not math.isfinite(t) or t <= 0.0:
        return 999.0
    return min(t, 999.0)


def kinematic_time_headway_s(distance_m, ego_speed_mag):
    if distance_m is None or ego_speed_mag is None:
        return 999.0
    try:
        dh = float(distance_m)
        ev = abs(float(ego_speed_mag))
    except (TypeError, ValueError):
        return 999.0
    if dh <= _DH_GAP_MIN or not math.isfinite(dh) or dh >= _DH_GAP_MAX:
        return 999.0
    if not math.isfinite(ev) or ev < _THW_DENOM_EPS:
        return 999.0
    th = dh / ev
    if not math.isfinite(th) or th <= 1e-4 or th >= 500.0:
        return 999.0
    return th


def _apply_tt_th_inv_to_row(row, dh_m, ego_speed_mag, lead_speed_mag):
    tt = kinematic_closing_ttc(dh_m, ego_speed_mag, lead_speed_mag)
    th = kinematic_time_headway_s(dh_m, ego_speed_mag)
    inv_ttc = reciprocal_inv_feature(tt)
    inv_th = reciprocal_inv_feature(th)
    row["ttc"] = _fmt_feat_scalar(tt)
    row["time_headway"] = _fmt_feat_scalar(th)
    row["inv_ttc"] = _fmt_inv_scalar(inv_ttc)
    row["inv_time_headway"] = _fmt_inv_scalar(inv_th)


# --- IDM -------------------------------------------------------------------------------------


def _idm_accel_scalar(v_ego, v_lead, gap_m, pm, eps_gap=1e-3):
    """Same law as ``fit_idm_per_driver.idm_accel_torch`` (scalar)."""
    v0 = float(pm["v0"])
    s0 = float(pm["s0"])
    a = float(pm["a"])
    b = float(pm["b"])
    T = float(pm["T"])
    delta = float(pm["delta"])
    v = float(v_ego)
    vl = float(v_lead)
    s = max(float(gap_m), eps_gap)
    dv = v - vl
    sab = math.sqrt(max(a * b, 1e-12))
    s_star = s0 + v * T + v * dv / (2.0 * sab)
    s_star = max(s_star, s0 + 1e-4)
    v0_safe = max(v0, eps_gap)
    ratio = v / v0_safe
    if ratio > 50.0:
        ratio = 50.0
    free_term = ratio ** delta
    acc = a * (1.0 - free_term - (s_star / s) ** 2)
    return acc


def _parse_follower_driver(s):
    return str(s).strip().upper()


def _driver_sort_key(di):
    if di.startswith("T") and di[1:].isdigit():
        return int(di[1:])
    return 9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_no_driver_idm",
    )
    ap.add_argument(
        "--idm_dir",
        type=str,
        required=True,
        help="Directory with <T*>/idm.json (fit_idm_per_driver output).",
    )
    ap.add_argument(
        "--follower_driver",
        type=str,
        required=True,
        help="Which subject's IDM params to load, e.g. T5 → <idm_dir>/T5/idm.json",
    )
    ap.add_argument("--lane_center_y", type=float, default=-7.625)
    ap.add_argument("--lane_width", type=float, default=3.75)
    ap.add_argument(
        "--lateral_jitter_limit_ratio",
        type=float,
        default=0.25,
    )
    ap.add_argument("--lateral_smooth_window", type=int, default=11)
    ap.add_argument("--lateral_mode", type=str, default="original_jitter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--warmup_frames",
        type=int,
        default=20,
        help="First N rows: ego accel = lead longitudinal accel (then IDM rollout).",
    )
    ap.add_argument(
        "--min_ego_speed_mps",
        type=float,
        default=0.15,
        help="Lower bound after each Euler velocity step (0 to disable clamp).",
    )
    ap.add_argument(
        "--pred_accel_clip_min",
        type=float,
        default=-8.0,
        help="Clip IDM acceleration to this minimum (m/s²).",
    )
    ap.add_argument(
        "--pred_accel_clip_max",
        type=float,
        default=6.0,
        help="Clip IDM acceleration to this maximum (m/s²).",
    )
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--lateral_pool_data_dir", type=str, default="")
    ap.add_argument("--lateral_pool_driver", type=str, default="")
    ap.add_argument("--lateral_pool_drivers", type=str, default="")

    args = ap.parse_args()
    follower = _parse_follower_driver(args.follower_driver)

    json_path = os.path.join(os.path.abspath(args.idm_dir), follower, "idm.json")
    if not os.path.isfile(json_path):
        raise SystemExit("Missing IDM params: {}".format(json_path))
    with open(json_path, "r", encoding="utf-8") as f:
        idm_blob = json.load(f)
    idm_params = idm_blob.get("parameters") or idm_blob
    for key in ("v0", "s0", "a", "b", "T", "delta"):
        if key not in idm_params:
            raise SystemExit("idm.json missing parameters.{!r}".format(key))

    random.seed(args.seed)
    scenario_paths = discover_scenario_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        scenario_paths = scenario_paths[: args.max_files]
    if not scenario_paths:
        print(
            "[WARN] No scenarios under {!r}: need driving_data.csv or segment_<n>.csv".format(
                args.data_dir
            ),
            file=sys.stderr,
        )
        return

    pool_root = args.lateral_pool_data_dir.strip() or args.data_dir
    pool_paths = discover_lateral_pool_csvs(pool_root)
    lat_drv = args.lateral_pool_driver.strip()
    lat_drvs_list = _parse_driver_id_list(args.lateral_pool_drivers)
    if lat_drvs_list:
        drv_set = set(lat_drvs_list)
        pool_paths = [p for p in pool_paths if _extract_driver_id(p) in drv_set]
    elif lat_drv:
        pool_paths = [p for p in pool_paths if _extract_driver_id(p) == lat_drv]
    if (
        not pool_paths
        and (lat_drv or lat_drvs_list)
        and args.lateral_mode in ("original_jitter", "pooled_mean_smooth")
    ):
        whom = ",".join(lat_drvs_list) if lat_drvs_list else lat_drv
        print(
            "[WARN] lateral pool drivers {}: no lateral pool CSVs under {}.".format(
                whom, pool_root
            )
        )

    os.makedirs(args.out_dir, exist_ok=True)
    lateral_pool = _build_driver_lateral_pool(pool_paths, args.lane_center_y)
    lateral_pool_drivers_disp = ",".join(lat_drvs_list) if lat_drvs_list else ""

    summary = []
    for fp in scenario_paths:
        rel_path = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        out_fp = os.path.join(args.out_dir, rel_path)
        out_parent = os.path.dirname(out_fp)
        if not os.path.isdir(out_parent):
            os.makedirs(out_parent)

        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not fieldnames or not rows:
            continue

        rows = [hydrate_following_row_aliases(dict(r)) for r in rows]
        targets = ["ego_a_long"]

        for k in targets:
            if k not in fieldnames:
                fieldnames.append(k)
        if "ego_acceleration" not in fieldnames:
            fieldnames.append("ego_acceleration")
        if "ego_a_long" not in fieldnames:
            fieldnames.append("ego_a_long")
        for _ek in ("ttc", "time_headway", "inv_ttc", "inv_time_headway"):
            if _ek not in fieldnames:
                fieldnames.append(_ek)

        ts = [_parse_float(r.get("timestamp")) for r in rows]
        n = len(rows)

        ex0 = _parse_float(rows[0].get("ego_pos_x"))
        lx0 = _parse_float(rows[0].get("lead_pos_x"))
        dh0 = _parse_float(rows[0].get("distance_headway"))
        distance_offset = None
        if ex0 is not None and lx0 is not None and dh0 is not None:
            distance_offset = dh0 - abs(ex0 - lx0)

        orig_long_acc = [_original_ego_long_accel(r) for r in rows]

        accel_sim = [0.0] * n
        ego_v_roll = [0.0] * n
        gap_roll = [0.0] * n

        for idx in range(n):
            ev = _row_value(rows[idx], "ego_v_long")
            dh = _row_value(rows[idx], "distance_headway")
            if ev is None or dh is None:
                raise RuntimeError(
                    "{} row {} missing ego_v_long or distance_headway".format(rel_path, idx)
                )
            accel_sim[idx] = _original_ego_long_accel(rows[idx])
            ego_v_roll[idx] = _bounded_posterior_speed(float(ev), args.min_ego_speed_mps)
            gap_roll[idx] = float(dh)

        first_model_i = max(0, int(args.warmup_frames))
        for i in range(first_model_i):
            la = (
                _lead_longitudinal_accel(rows[i])
                if i < args.warmup_frames
                else orig_long_acc[i]
            )
            la_s = "{:.6f}".format(la)
            if "ego_a_long" in rows[i]:
                rows[i]["ego_a_long"] = la_s
            if "ego_acceleration" in rows[i]:
                rows[i]["ego_acceleration"] = la_s
            accel_sim[i] = float(la)
            if i + 1 < n:
                ti, tj = ts[i], ts[i + 1]
                if ti is not None and tj is not None:
                    dt_w = max(0.0, float(tj) - float(ti))
                    ego_v_roll[i + 1] = _bounded_posterior_speed(
                        ego_v_roll[i] + float(la) * dt_w,
                        args.min_ego_speed_mps,
                    )

        n_pred = 0
        skip_idm = 0
        for t in range(first_model_i, n):
            lv_e = _row_value(rows[t], "lead_v_long")
            if lv_e is None:
                lv_e = _parse_float(rows[t].get("lead_speed"))
            if lv_e is None:
                skip_idm += 1
                if t + 1 < n and ts[t] is not None and ts[t + 1] is not None:
                    dt_carry = max(0.0, float(ts[t + 1]) - float(ts[t]))
                    ego_v_roll[t + 1] = _bounded_posterior_speed(
                        ego_v_roll[t],
                        args.min_ego_speed_mps,
                    )
                    lv_carry = _row_value(rows[t], "lead_v_long")
                    if lv_carry is None:
                        lv_carry = _parse_float(rows[t].get("lead_speed"))
                    if lv_carry is not None:
                        gap_roll[t + 1] = gap_roll[t] + (
                            float(lv_carry) - float(ego_v_roll[t])
                        ) * dt_carry
                continue
            lv_use = float(lv_e)

            gh = gap_roll[t]
            vh = ego_v_roll[t]
            pa_raw = _idm_accel_scalar(vh, lv_use, gh, idm_params)
            pa = _clip_prediction_accel(
                pa_raw, args.pred_accel_clip_min, args.pred_accel_clip_max
            )
            accel_sim[t] = pa
            if "ego_a_long" in rows[t]:
                rows[t]["ego_a_long"] = "{:.6f}".format(pa)
            if "ego_acceleration" in rows[t]:
                rows[t]["ego_acceleration"] = "{:.6f}".format(pa)

            if t + 1 < n:
                if ts[t + 1] is None or ts[t] is None:
                    raise RuntimeError(
                        "{} timestamps missing during IDM rollout at {}".format(rel_path, t)
                    )
                dt_fwd = max(0.0, ts[t + 1] - ts[t])
                lv_fwd = lv_use
                ego_v_roll[t + 1] = _bounded_posterior_speed(
                    ego_v_roll[t] + pa * dt_fwd,
                    args.min_ego_speed_mps,
                )
                gap_roll[t + 1] = gap_roll[t] + (lv_fwd - ego_v_roll[t]) * dt_fwd

            ego_v_safe = max(0.0, float(ego_v_roll[t]))
            lv_f = max(0.0, lv_use)
            rel_spd = lv_f - ego_v_safe
            gf = float(gap_roll[t])
            tc = kinematic_closing_ttc(gf, ego_v_safe, lv_f)
            th = kinematic_time_headway_s(gf, ego_v_safe)
            rr = rows[t]

            rr["distance_headway"] = _fmt_feat_scalar(gf)
            rr["ego_a_long"] = _fmt_feat_scalar(pa)
            rr["ego_acceleration"] = _fmt_feat_scalar(pa)
            rr["relative_v_long"] = _fmt_feat_scalar(rel_spd)
            if "relative_speed" in rr:
                rr["relative_speed"] = _fmt_feat_scalar(rel_spd)
            rr["ttc"] = _fmt_feat_scalar(tc)
            rr["time_headway"] = _fmt_feat_scalar(th)
            rr["inv_ttc"] = _fmt_inv_scalar(reciprocal_inv_feature(tc))
            rr["inv_time_headway"] = _fmt_inv_scalar(reciprocal_inv_feature(th))

            n_pred += 1

        if skip_idm:
            print("[WARN] {} rows skipped (missing lead velocity): {}".format(rel_path, skip_idm))

        _finalize_longitudinal_from_roll(
            rows, ego_v_roll, distance_offset, args.min_ego_speed_mps
        )

        if args.lateral_mode == "original_jitter":
            lat_id = lat_drv or _extract_driver_id(fp)
            if lat_drvs_list:
                if len(lat_drvs_list) > 1 and not lat_drv:
                    print(
                        "[WARN] lateral_pool_driver unset with multiple lateral_pool_drivers {}; "
                        "using {}.".format(
                            lateral_pool_drivers_disp,
                            sorted(lat_drvs_list, key=_driver_sort_key)[0],
                        )
                    )
                if not lat_drv:
                    lat_id = sorted(lat_drvs_list, key=_driver_sort_key)[0]
            p = lateral_pool.get(lat_id, {})
            if not p.get("y_res") and lat_id != _extract_driver_id(fp):
                p = lateral_pool.get(_extract_driver_id(fp), {})
            rng = random.Random(args.seed + abs(hash(rel_path)) % 1000003)
            y_res = _sample_with_wrap(p.get("y_res", []), len(rows), rng)
            yaw_res = _sample_with_wrap(p.get("yaw_res", []), len(rows), rng)
            steer_res = _sample_with_wrap(p.get("steer_res", []), len(rows), rng)
            lateral_limit = args.lane_width * args.lateral_jitter_limit_ratio
            y_res, yaw_res, steer_res, lateral_source = _smooth_lateral_if_needed(
                y_res,
                yaw_res,
                steer_res,
                lateral_limit,
                args.lateral_smooth_window,
            )
            for i in range(len(rows)):
                rows[i]["ego_pos_y"] = "{:.6f}".format(args.lane_center_y + y_res[i])
                if "ego_yaw" in rows[i]:
                    rows[i]["ego_yaw"] = "{:.6f}".format(yaw_res[i])
                if "steer" in rows[i]:
                    rows[i]["steer"] = "{:.6f}".format(steer_res[i])
        elif args.lateral_mode == "pooled_mean_smooth":
            drv_for_pool = lat_drvs_list[:] if lat_drvs_list else ([lat_drv] if lat_drv else [])
            drv_for_pool = sorted(drv_for_pool, key=_driver_sort_key)
            if drv_for_pool:
                y_cat, yaw_cat, steer_cat = _concat_pooled_residuals(
                    lateral_pool, drv_for_pool
                )
                smooth_y = _moving_average(y_cat, args.lateral_smooth_window)
                smooth_yaw = _moving_average(yaw_cat, args.lateral_smooth_window)
                smooth_steer = _moving_average(steer_cat, args.lateral_smooth_window)
                lateral_limit = args.lane_width * args.lateral_jitter_limit_ratio
                y_res, yaw_res, steer_res, sm_tag = _smooth_lateral_if_needed(
                    smooth_y,
                    smooth_yaw,
                    smooth_steer,
                    lateral_limit,
                    args.lateral_smooth_window,
                )
                lateral_source = (
                    "pooled_mean_smooth_limited"
                    if sm_tag == "driver_residual_pool_smoothed_limited"
                    else "pooled_mean_smooth"
                )
                y_res, yaw_res, steer_res = _residuals_to_length(
                    y_res, yaw_res, steer_res, len(rows)
                )
                for i in range(len(rows)):
                    rows[i]["ego_pos_y"] = "{:.6f}".format(args.lane_center_y + y_res[i])
                    if "ego_yaw" in rows[i]:
                        rows[i]["ego_yaw"] = "{:.6f}".format(yaw_res[i])
                    if "steer" in rows[i]:
                        rows[i]["steer"] = "{:.6f}".format(steer_res[i])
            else:
                lateral_source = "original_rows_pool_mean_smooth_no_drivers"
                print(
                    "[WARN] pooled_mean_smooth needs --lateral_pool_drivers or --lateral_pool_driver; "
                    "keeping lateral for {}.".format(rel_path)
                )
        else:
            lateral_source = "original_rows"

        with open(out_fp, "w", encoding="utf-8", newline="") as wf:
            w = csv.DictWriter(wf, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        summary.append(
            {
                "source_file": rel_path,
                "out_file": os.path.relpath(out_fp, args.out_dir).replace("\\", "/"),
                "n_rows": str(len(rows)),
                "n_pred_rows": str(n_pred),
                "follower_driver": follower,
                "idm_json": json_path,
                "lateral_mode": args.lateral_mode,
                "lateral_pool_driver": lat_drv,
                "lateral_pool_drivers": lateral_pool_drivers_disp,
                "lateral_source": lateral_source,
                "longitudinal_model": "idm",
            }
        )

    sum_fp = os.path.join(args.out_dir, "generation_summary.csv")
    with open(sum_fp, "w", encoding="utf-8", newline="") as sf:
        fields = summary[0].keys() if summary else []
        w = csv.DictWriter(sf, fieldnames=list(fields))
        if summary:
            w.writeheader()
            for r in summary:
                w.writerow(r)

    print("[OK] generated:", len(summary), "written to", args.out_dir)
    if summary:
        print("[OK] summary:", sum_fp)


if __name__ == "__main__":
    main()
