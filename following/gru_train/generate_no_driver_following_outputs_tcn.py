# -*- coding: utf-8 -*-
"""
Generate no-driver-intervention following outputs using trained **BC-TCN** (from ``train_bc_tcn.py``).

Same closed-loop pipeline as ``generate_no_driver_following_outputs.py`` (causal window, warm-up,
integration, lateral jitter / pooled residual). Requires ``model_meta.json`` with ``tcn_channels``
(from ``train_bc_tcn.py``); ``tcn_kernel_size`` defaults to 3 if omitted; ``arch`` should be ``tcn``.

COMMON_CASE="/home/zwx/driver_model/following/outputs/following_calibrated/T12/行车/20260421_120610_198_exp1_f"

for i in $(seq 1 20); do
  D="T${i}"
  python3 /home/zwx/driver_model/following/gru_train/generate_no_driver_following_outputs_tcn.py \
    --data_dir "${COMMON_CASE}" \
    --model_dir "/home/zwx/driver_model/following/outputs/il_bc_tcn_per_driver/${D}" \
    --out_dir "/home/zwx/driver_model/following/outputs/personalized_no_driver_tcn_common_lead/${D}" \
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from bc_gru_features import (
    DEFAULT_FEATURES,
    DEFAULT_FEATURES_LEGACY,
    _parse_float,
    _row_value,
    hydrate_bc_gru_row_aliases,
    reciprocal_inv_feature,
    scalar_features_for_row,
)


def _integration_timestamps(rows):
    """Prefer ``sim_time_s`` (uniform CARLA step); else ``timestamp``."""
    sts = [_parse_float(r.get("sim_time_s")) for r in rows]
    if sts and all(x is not None for x in sts):
        return sts
    return [_parse_float(r.get("timestamp")) for r in rows]


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
    """
    Scenario trajectories for rollout: ``driving_data.csv`` under allowed trees if present,
    otherwise ``segment_<n>.csv`` (same path filters as ``discover_following_csvs``).
    """
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
    """All ``segment_NNN.csv`` under data_dir (e.g. imitation-learning clean output)."""
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def discover_lateral_pool_csvs(data_dir):
    """Paths usable for lateral residual pools: IL segments and/or following driving_data.csv."""
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
    """Concatenate y/yaw/steer residuals for drivers in order (sorted T*)."""

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
    """Deterministic cycling: index i -> smooth[i % L]."""
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
    """Nonnegative speed optional floor (``vmin_mps``>0 avoids BC 'sticking' at zero in export)."""
    x = max(0.0, float(v_euler_end))
    if vmin_mps is not None and float(vmin_mps) > 0.0:
        x = max(x, float(vmin_mps))
    return x


def _clip_prediction_accel(pa_raw, lo, hi):
    """Clamp predicted longitudinal accel to [lo, hi] (handles swapped bounds)."""
    a = float(lo)
    b = float(hi)
    if a > b:
        a, b = b, a
    return max(a, min(b, float(pa_raw)))


def _finalize_longitudinal_from_roll(rows, ego_v_roll, distance_offset, vmin_mps):
    """Write ``ego_speed``/``ego_v_long`` from closed-loop ``ego_v_roll``; integrate ``ego_pos_x`` (subtract path)."""
    if not rows:
        return
    if len(ego_v_roll) != len(rows):
        raise RuntimeError(
            "ego_v_roll length {} != rows {}".format(len(ego_v_roll), len(rows))
        )
    n = len(rows)
    ts = _integration_timestamps(rows)
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
    """Replay the driver's original lateral residuals, but damp unsafe swings.

    If the lateral residual exceeds the configured fraction of lane width, smooth the
    residual sequence and clip it to the limit so generated trajectories stay near the lane center.
    """
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
    """Match calibrated / IL CSV precision for reciprocal channels."""
    return "{:.9f}".format(float(x))


_THW_DENOM_EPS = 0.5  # aligns with following/scripts/calibrate_following_data._THW_DENOM_EPS
_DH_GAP_MIN = 1e-3
_DH_GAP_MAX = 300.0


def kinematic_closing_ttc(distance_m, ego_speed_mag, lead_speed_mag):
    """
    Match ``calibrate_following_data._longitudinal_closing_ttc_s``::
    closing when ``ego_speed - lead_speed > 1e-2``, else sentinel 999.
    """
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
    """Same bounds as ``calibrate_following_data._longitudinal_time_headway_s`` (positive speed magnitude)."""
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
    """Writes ``ttc`` / ``time_headway`` / ``inv_*`` from scalar closing kinematics (mutates ``row``)."""
    tt = kinematic_closing_ttc(dh_m, ego_speed_mag, lead_speed_mag)
    th = kinematic_time_headway_s(dh_m, ego_speed_mag)
    inv_ttc = reciprocal_inv_feature(tt)
    inv_th = reciprocal_inv_feature(th)
    row["ttc"] = _fmt_feat_scalar(tt)
    row["time_headway"] = _fmt_feat_scalar(th)
    row["inv_ttc"] = _fmt_inv_scalar(inv_ttc)
    row["inv_time_headway"] = _fmt_inv_scalar(inv_th)


def build_closed_loop_feature_row(template_row, ego_v_f, accel_in_f, gap_f):
    """
    Shallow-copy template lead columns; override ego/longitudinal-gap fields consumed by BC policies.

    Closing kinematics align with ``calibrate_following_data`` (scalar ``ego − lead``
    closing rate; bounded gap / THW thresholds). Writes ``inv_ttc`` / ``inv_time_headway``
    (0 when ``ttc≈999`` / invalid).
    """
    r = dict(template_row)
    ego_v_safe = max(0.0, float(ego_v_f))
    lv_raw = _row_value(template_row, "lead_v_long")
    if lv_raw is None:
        lv_raw = _parse_float(template_row.get("lead_speed"))
    if lv_raw is None:
        return None
    lv_f = max(0.0, float(lv_raw))

    rel = lv_f - ego_v_safe
    accel_f = float(accel_in_f)
    gf = float(gap_f)

    tc = kinematic_closing_ttc(gf, ego_v_safe, lv_f)
    th = kinematic_time_headway_s(gf, ego_v_safe)
    inv_ttc = reciprocal_inv_feature(tc)
    inv_th = reciprocal_inv_feature(th)

    r["ego_a_long"] = _fmt_feat_scalar(accel_f)
    r["ego_acceleration"] = _fmt_feat_scalar(accel_f)
    r["distance_headway"] = _fmt_feat_scalar(gf)
    r["relative_v_long"] = _fmt_feat_scalar(rel)
    if "relative_speed" in r:
        r["relative_speed"] = _fmt_feat_scalar(rel)
    r["ttc"] = _fmt_feat_scalar(tc)
    r["time_headway"] = _fmt_feat_scalar(th)
    r["inv_ttc"] = _fmt_inv_scalar(inv_ttc)
    r["inv_time_headway"] = _fmt_inv_scalar(inv_th)
    return r


def _accel_feat_for_closed_loop_window(idx, t_pred, accel_sim):
    """Training-time leakage fix: timestep ``t_pred`` uses ``accel_sim[t_pred-1]`` as ego_a_long input."""
    if idx < t_pred:
        return accel_sim[idx]
    if idx == t_pred:
        return accel_sim[t_pred - 1]
    raise RuntimeError("closed-loop window idx out of range")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/home/zwx/driver_model/data")
    ap.add_argument(
        "--model_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/il_bc_tcn_per_driver/T9",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_no_driver_bc_tcn",
    )
    ap.add_argument("--lane_center_y", type=float, default=-7.625)
    ap.add_argument("--lane_width", type=float, default=3.75)
    ap.add_argument(
        "--lateral_jitter_limit_ratio",
        type=float,
        default=0.25,
        help="Smooth/limit lateral residuals when abs(residual) exceeds this fraction of lane width",
    )
    ap.add_argument(
        "--lateral_smooth_window",
        type=int,
        default=11,
        help="Moving-average window used when lateral residuals exceed the limit",
    )
    ap.add_argument("--lateral_mode", type=str, default="original_jitter")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--warmup_frames",
        type=int,
        default=20,
        help="First N rows: set ego longitudinal acceleration to match lead; model starts at max(warmup_frames, seq_len) (causal window).",
    )
    ap.add_argument(
        "--min_ego_speed_mps",
        type=float,
        default=0.15,
        help="Clamp closed-loop/export ego speed ≥ this (m/s) after each Euler step. "
        "Default 0.15 mitigates low-speed rollout where BC keeps braking to zero; use 0 to disable.",
    )
    ap.add_argument(
        "--pred_accel_clip_min",
        type=float,
        default=-8.0,
        help="Clamp BC-predicted ego longitudinal acceleration ≥ this value (m/s²); align with calibrated data.",
    )
    ap.add_argument(
        "--pred_accel_clip_max",
        type=float,
        default=6.0,
        help="Clamp BC-predicted ego longitudinal acceleration ≤ this value (m/s²); align with calibrated data.",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument(
        "--lateral_pool_data_dir",
        type=str,
        default="",
        help="If set, build lateral residual pool only from segment_*.csv under this tree (default: same as --data_dir).",
    )
    ap.add_argument(
        "--lateral_pool_driver",
        type=str,
        default="",
        help="Restrict lateral pool CSVs to this driver. Ignored when --lateral_pool_drivers is non-empty.",
    )
    ap.add_argument(
        "--lateral_pool_drivers",
        type=str,
        default="",
        help="Comma-separated driver ids (e.g. T2,T9,T16). When set, only these drivers contribute to "
        "the lateral pool CSV set. Required for pooled_mean_smooth (defines who is merged/smoothed).",
    )
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch import nn

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
            super(CausalConv1d, self).__init__()
            self._pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=0, dilation=dilation,
            )

        def forward(self, x):
            if self._pad > 0:
                x = F.pad(x, (self._pad, 0))
            return self.conv(x)

    class TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
            super(TemporalBlock, self).__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(dropout)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.drop1(self.relu1(self.conv1(x)))
            out = self.drop2(self.relu2(self.conv2(out)))
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class BCTCN(nn.Module):
        """Must stay in sync with ``train_bc_tcn.BCTCN``."""

        def __init__(self, input_dim, channel_dims, kernel_size, dropout, output_dim):
            super(BCTCN, self).__init__()
            if not channel_dims:
                raise ValueError("channel_dims must be non-empty")
            layers = []
            for i, out_c in enumerate(channel_dims):
                dilation = 2 ** i
                in_c = input_dim if i == 0 else channel_dims[i - 1]
                layers.append(TemporalBlock(in_c, out_c, kernel_size, dilation, dropout))
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Linear(channel_dims[-1], output_dim)

        def forward(self, x):
            z = x.transpose(1, 2)
            z = self.tcn(z)
            last = z[:, :, -1]
            return self.head(last)

    report_path = os.path.join(args.model_dir, "train_report.json")
    meta_path = os.path.join(args.model_dir, "model_meta.json")
    model_path = os.path.join(args.model_dir, "best_model.pt")
    for label, p in (
        ("train_report.json", report_path),
        ("model_meta.json", meta_path),
        ("best_model.pt", model_path),
    ):
        if not os.path.isfile(p):
            raise SystemExit(
                "Missing {!r}:\n  {}\n"
                "This directory is not a complete train_bc_tcn.py output. "
                "Train that driver (or fix the path), then rerun.".format(label, p)
            )
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = list(meta["features"])
    targets = list(meta["targets"])
    seq_len = int(meta["seq_len"])
    if list(features) not in (
        list(DEFAULT_FEATURES),
        list(DEFAULT_FEATURES_LEGACY),
    ):
        print(
            "[WARN] model_meta.json feature list differs from bc_gru_features "
            "DEFAULT_FEATURES / DEFAULT_FEATURES_LEGACY (order or names mismatch — "
            "closed loop may diverge from training)."
        )
    feat_mean = np.asarray(report["feature_mean"], dtype=np.float32)
    feat_std = np.asarray(report["feature_std"], dtype=np.float32)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    arch = str(meta.get("arch", "")).strip().lower()
    if arch and arch != "tcn":
        print(
            "[WARN] model_meta.json arch={!r} is not 'tcn'; this script expects a TCN checkpoint.".format(
                meta.get("arch")
            ),
            file=sys.stderr,
        )
    if "tcn_channels" not in meta:
        raise SystemExit(
            "model_meta.json missing 'tcn_channels' (need train_bc_tcn.py export). Got keys: {}".format(
                sorted(meta.keys())
            )
        )
    ch_meta = meta["tcn_channels"]
    if isinstance(ch_meta, list):
        channel_dims = [int(x) for x in ch_meta]
    else:
        channel_dims = [
            int(x.strip()) for x in str(ch_meta).split(",") if str(x).strip()
        ]
    if not channel_dims:
        raise SystemExit(
            "model_meta.json 'tcn_channels' parsed to an empty list (check value: {!r}).".format(
                ch_meta
            )
        )
    tcn_kernel_size = int(meta.get("tcn_kernel_size", 3))
    dropout_inf = float(meta.get("dropout", 0.1))

    model = BCTCN(
        input_dim=len(features),
        channel_dims=channel_dims,
        kernel_size=tcn_kernel_size,
        dropout=dropout_inf,
        output_dim=len(targets),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scenario_paths = discover_scenario_csvs(args.data_dir)
    if args.max_files and args.max_files > 0:
        scenario_paths = scenario_paths[: args.max_files]
    if not scenario_paths:
        print(
            "[WARN] No scenarios under {!r}: need driving_data.csv or segment_<n>.csv "
            "(path must match following/_f session filters). Exit.".format(args.data_dir)
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
            "[WARN] lateral pool drivers {}: no lateral pool CSVs "
            "(segment_*.csv or following driving_data.csv) under {}.".format(whom, pool_root)
        )
    os.makedirs(args.out_dir, exist_ok=True)
    lateral_pool = _build_driver_lateral_pool(pool_paths, args.lane_center_y)

    summary = []
    for fp in scenario_paths:
        rel = os.path.relpath(fp, args.data_dir).replace("\\", "/")
        out_fp = os.path.join(args.out_dir, rel)
        out_parent = os.path.dirname(out_fp)
        if not os.path.isdir(out_parent):
            os.makedirs(out_parent)

        with open(fp, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if not fieldnames or not rows:
            continue

        rows = [hydrate_bc_gru_row_aliases(dict(r)) for r in rows]

        for k in targets:
            if k not in fieldnames:
                fieldnames.append(k)
        if "ego_a_long" in targets and "ego_acceleration" not in fieldnames:
            fieldnames.append("ego_acceleration")
        if "ego_acceleration" in targets and "ego_a_long" not in fieldnames:
            fieldnames.append("ego_a_long")
        for _ek in ("ttc", "time_headway", "inv_ttc", "inv_time_headway"):
            if _ek not in fieldnames:
                fieldnames.append(_ek)

        ex0 = _parse_float(rows[0].get("ego_pos_x"))
        lx0 = _parse_float(rows[0].get("lead_pos_x"))
        dh0 = _parse_float(rows[0].get("distance_headway"))
        distance_offset = None
        if ex0 is not None and lx0 is not None and dh0 is not None:
            distance_offset = dh0 - abs(ex0 - lx0)

        orig_long_acc = [_original_ego_long_accel(r) for r in rows]

        ts = _integration_timestamps(rows)
        n = len(rows)
        accel_sim = [0.0] * n
        ego_v_roll = [0.0] * n
        gap_roll = [0.0] * n

        for idx in range(n):
            ev = _row_value(rows[idx], "ego_v_long")
            dh = _row_value(rows[idx], "distance_headway")
            if ev is None or dh is None:
                raise RuntimeError(
                    "{} row {} missing ego_v_long or distance_headway (needed for bc_gru_features)".format(
                        rel, idx,
                    )
                )
            accel_sim[idx] = _original_ego_long_accel(rows[idx])
            ego_v_roll[idx] = _bounded_posterior_speed(float(ev), args.min_ego_speed_mps)
            gap_roll[idx] = float(dh)

        first_model_i = max(args.warmup_frames, seq_len)
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
            # Keep closed-loop speed consistent with warmup accelerations before the model steps in.
            if i + 1 < n:
                ti, tj = ts[i], ts[i + 1]
                if ti is not None and tj is not None:
                    dt_w = max(0.0, float(tj) - float(ti))
                    ego_v_roll[i + 1] = _bounded_posterior_speed(
                        ego_v_roll[i] + float(la) * dt_w,
                        args.min_ego_speed_mps,
                    )

        n_pred = 0
        n_feat_skip = 0
        for t in range(first_model_i, n):
            feats_win = []
            ok_w = True
            # Causal window: rows [t-seq_len, t-1] only (matches train_bc_gru / train_bc_tcn sampling).
            for j in range(t - seq_len, t):
                if j < first_model_i:
                    row_eff = rows[j]
                else:
                    a_in = _accel_feat_for_closed_loop_window(j, t, accel_sim)
                    sr = build_closed_loop_feature_row(
                        rows[j], ego_v_roll[j], a_in, gap_roll[j],
                    )
                    if sr is None:
                        ok_w = False
                        break
                    row_eff = sr
                if "dt_prev" in features:
                    if j == 0:
                        dt_prev = 0.0
                    else:
                        t0, t1 = ts[j - 1], ts[j]
                        if t0 is None or t1 is None:
                            ok_w = False
                            break
                        dt_prev = max(0.0, t1 - t0)
                else:
                    dt_prev = 0.0
                fv = scalar_features_for_row(row_eff, dt_prev, features)
                if fv is None:
                    ok_w = False
                    break
                feats_win.append(np.asarray(fv, dtype=np.float32))

            if not ok_w:
                n_feat_skip += 1
                if t + 1 < n and ts[t] is not None and ts[t + 1] is not None:
                    dt_carry = max(0.0, ts[t + 1] - ts[t])
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

            x = np.stack(feats_win)
            x = (x - feat_mean) / feat_std
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                yp = model(xt).cpu().numpy()[0]

            try:
                pa_i = targets.index("ego_a_long")
            except ValueError:
                try:
                    pa_i = targets.index("ego_acceleration")
                except ValueError:
                    pa_i = 0
            pa_raw = float(yp[pa_i]) if len(yp) else 0.0
            pa = _clip_prediction_accel(
                pa_raw, args.pred_accel_clip_min, args.pred_accel_clip_max
            )
            for j, kt in enumerate(targets):
                vj = float(yp[j])
                if kt in ("ego_a_long", "ego_acceleration"):
                    vj = pa
                rows[t][kt] = "{:.6f}".format(vj)
                if kt == "ego_a_long" and "ego_acceleration" in rows[t]:
                    rows[t]["ego_acceleration"] = "{:.6f}".format(pa)
                elif kt == "ego_acceleration" and "ego_a_long" in rows[t]:
                    rows[t]["ego_a_long"] = "{:.6f}".format(pa)
            accel_sim[t] = pa

            if t + 1 < n:
                if ts[t + 1] is None or ts[t] is None:
                    raise RuntimeError(
                        "{} timestamps missing during closed-loop rollout at {}".format(rel, t)
                    )
                dt_fwd = max(0.0, ts[t + 1] - ts[t])
                lv_t = _row_value(rows[t], "lead_v_long")
                if lv_t is None:
                    lv_t = _parse_float(rows[t].get("lead_speed"))
                if lv_t is None:
                    raise RuntimeError(
                        "{} row {} missing lead_v_long and lead_speed (rolling gap)".format(
                            rel,
                            t,
                        )
                    )
                ego_v_roll[t + 1] = _bounded_posterior_speed(
                    ego_v_roll[t] + pa * dt_fwd,
                    args.min_ego_speed_mps,
                )
                gap_roll[t + 1] = gap_roll[t] + (float(lv_t) - ego_v_roll[t]) * dt_fwd

            n_pred += 1

        if n_feat_skip:
            print(
                "[WARN] {} closed-loop steps skipped (invalid features / lead_v_long); "
                "kept CSV longitudinal for those rows.".format(rel, n_feat_skip)
            )

        _finalize_longitudinal_from_roll(
            rows, ego_v_roll, distance_offset, args.min_ego_speed_mps
        )

        lateral_pool_drivers_disp = ",".join(lat_drvs_list) if lat_drvs_list else ""

        if args.lateral_mode == "original_jitter":
            lat_id = lat_drv or _extract_driver_id(fp)
            if lat_drvs_list:
                if len(lat_drvs_list) > 1 and not lat_drv:
                    print(
                        "[WARN] lateral_pool_driver unset with multiple lateral_pool_drivers {}; "
                        "using {} for jitter sample order.".format(
                            lateral_pool_drivers_disp,
                            sorted(lat_drvs_list, key=lambda d: int(d[1:]) if d.startswith("T") and d[1:].isdigit() else 9999)[0],
                        )
                    )
                if not lat_drv:
                    lat_id = sorted(
                        lat_drvs_list,
                        key=lambda d: int(d[1:]) if d.startswith("T") and d[1:].isdigit() else 9999,
                    )[0]
            p = lateral_pool.get(lat_id, {})
            if not p.get("y_res") and lat_id != _extract_driver_id(fp):
                p = lateral_pool.get(_extract_driver_id(fp), {})
            rng = random.Random(args.seed + abs(hash(rel)) % 1000003)
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
            drv_for_pool = sorted(
                drv_for_pool,
                key=lambda d: int(d[1:])
                if d.startswith("T") and d[1:].isdigit()
                else 9999,
            )
            if drv_for_pool:
                y_cat, yaw_cat, steer_cat = _concat_pooled_residuals(lateral_pool, drv_for_pool)
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
                y_res, yaw_res, steer_res = _residuals_to_length(y_res, yaw_res, steer_res, len(rows))
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
                    "keeping original lateral columns for {}.".format(rel)
                )
        else:
            lateral_source = "original_rows"

        with open(out_fp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        summary.append(
            {
                "source_file": rel,
                "out_file": os.path.relpath(out_fp, args.out_dir).replace("\\", "/"),
                "n_rows": str(len(rows)),
                "n_pred_rows": str(n_pred),
                "driver_id": _extract_driver_id(fp),
                "policy_arch": "tcn",
                "lateral_mode": args.lateral_mode,
                "lateral_pool_driver": lat_drv,
                "lateral_pool_drivers": lateral_pool_drivers_disp,
                "lateral_source": lateral_source,
            }
        )

    sum_fp = os.path.join(args.out_dir, "generation_summary.csv")
    with open(sum_fp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "out_file",
                "n_rows",
                "n_pred_rows",
                "driver_id",
                "policy_arch",
                "lateral_mode",
                "lateral_pool_driver",
                "lateral_pool_drivers",
                "lateral_source",
            ],
        )
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print("[OK] generated files:", len(summary))
    print("[OK] summary:", sum_fp)


if __name__ == "__main__":
    main()

