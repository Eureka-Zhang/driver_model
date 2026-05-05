# -*- coding: utf-8 -*-
"""
Fit a classic Intelligent Driver Model (Treiber-Hennecke-Helbing) per subject from
following trajectories: ``**/segment_<n>.csv`` (imitation-clean) or ``**/driving_data.csv``
(calibrated sessions). Rows are sorted by ``sim_time_s`` when present (--time-column),
else ``timestamp``, before pooling samples.

Uses the textbook acceleration law (same unit convention as calibrated IL CSV):

  a_IDM = a * [ 1 - (v/v0)**delta - (s*/s)**2 ]

  s* = s0 + v*T + v*(v - v_lead) / (2 * sqrt(a*b))

State at each timestep: values read via local ``hydrate_following_row`` / ``_row_value_csv``
(no dependency on BC-GRU training code).

Optimization: constrained reparameterisation + Adam in PyTorch (no SciPy dependency).

Output (per driver + summary):

  <out_dir>/idm_all_drivers.json
  <out_dir>/idm_per_driver_summary.csv
  <out_dir>/<T*>/idm.json

Batch example::

  python3 /home/zwx/driver_model/following/idm/fit_idm_per_driver.py \
    --data_dir /home/zwx/driver_model/following/outputs/following_calibrated \
    --out_dir /home/zwx/driver_model/following/outputs/idm_per_driver \
    --n_restarts 5 --epochs 3000

Restrict drivers::

    --drivers T1,T3,T11
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
import torch

# --- CSV helpers (standalone; semantics aligned with calibrated IL segments) -------------
_SENTINEL_999_TOL = 1e-3


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


def _reciprocal_thw_like(x):
    """1/x for finite x > 0 and not ~999; else 0 (matches imitation-clean reciprocals)."""
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


def _row_value_csv(row, key):
    """Read one scalar column; mirrors common following CSV aliases."""
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
            return _reciprocal_thw_like(tt)
        return None

    if key == "inv_time_headway":
        vu = _parse_float(row.get("inv_time_headway"))
        if vu is not None:
            return float(vu)
        th = _parse_float(row.get("time_headway"))
        if th is not None:
            return _reciprocal_thw_like(th)
        return None

    if key == "ttc_inverse":
        return _row_value_csv(row, "inv_ttc")
    if key == "time_headway_inverse":
        return _row_value_csv(row, "inv_time_headway")

    if key in row:
        return _parse_float(row.get(key))
    return None


def hydrate_following_row(row):
    """
    Insert canonical longitudinal fields when CSV only has aliases (ego_speed → ego_v_long, etc.).
    """
    r = dict(row)

    def _pf(key):
        if key not in r:
            return None
        return _parse_float(r.get(key))

    def _missing(key):
        return _pf(key) is None

    if _missing("ego_v_long"):
        for s in ("ego_speed", "ego_v"):
            if _pf(s) is not None:
                r["ego_v_long"] = str(r[s]).strip()
                break

    if _missing("ego_a_long"):
        for s in ("ego_acceleration", "ego_accel", "accel_long"):
            if _pf(s) is not None:
                r["ego_a_long"] = str(r[s]).strip()
                break

    if _missing("lead_v_long"):
        for s in ("lead_speed",):
            if _pf(s) is not None:
                r["lead_v_long"] = str(r[s]).strip()
                break

    if _missing("relative_v_long"):
        if _pf("relative_speed") is not None:
            r["relative_v_long"] = str(r["relative_speed"]).strip()
        elif _pf("lead_v_long") is not None and _pf("ego_v_long") is not None:
            r["relative_v_long"] = "{:.6f}".format(
                float(_pf("lead_v_long")) - float(_pf("ego_v_long"))
            )

    if _missing("time_headway"):
        dh = _pf("distance_headway")
        ev = _pf("ego_v_long")
        if dh is not None and ev is not None:
            if float(ev) < 0.5:
                r["time_headway"] = "999.0"
            else:
                r["time_headway"] = "{:.6f}".format(float(dh) / float(ev))

    if _missing("ttc"):
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

    if _missing("inv_ttc"):
        tt = _pf("ttc")
        if tt is not None:
            r["inv_ttc"] = "{:.9f}".format(_reciprocal_thw_like(tt))

    if _missing("inv_time_headway"):
        th = _pf("time_headway")
        if th is not None:
            r["inv_time_headway"] = "{:.9f}".format(_reciprocal_thw_like(th))

    return r


def _discover_following_csvs(data_dir):
    """Collect ``segment_<n>.csv`` (IL-clean) or ``driving_data.csv`` (calibrated exports)."""
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if fn == "driving_data.csv" or re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _sort_rows_by_time(rows, primary, fallback):
    """Chronological order for reproducibility; IDM residual is instantaneous but avoids shuffled CSVs."""
    ts = [_parse_float(r.get(primary)) for r in rows]
    if ts and all(t is not None for t in ts):
        order = sorted(range(len(rows)), key=lambda i: ts[i])
        return [rows[i] for i in order]
    ts2 = [_parse_float(r.get(fallback)) for r in rows]
    if ts2 and all(t is not None for t in ts2):
        order = sorted(range(len(rows)), key=lambda i: ts2[i])
        return [rows[i] for i in order]
    return rows


def _extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _parse_ids(s):
    if not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def idm_accel_torch(v, vl, gap, v0, s0, a, b, T, delta, eps_gap=1e-3):
    """
    Torch vectors; differentiable. ``v``, ``vl``, ``gap`` same shape."""
    vv = v
    s = torch.clamp(gap, min=eps_gap)
    dv = vv - vl
    sab = torch.sqrt(torch.clamp(a * b, min=1e-12))
    s_star = s0 + vv * T + vv * dv / (2.0 * sab)
    s_star = torch.clamp(s_star, min=s0 + 1e-4)
    v0_safe = torch.clamp(v0, min=eps_gap)
    free_term = torch.pow(torch.clamp(vv / v0_safe, max=50.0), delta)
    acc = a * (1.0 - free_term - torch.pow(s_star / s, 2))
    return acc


def _box_params(raw, bounds):
    """raw: (..., n) unconstrained reals -> bounded tensors."""
    out = []
    for i, (lo, hi) in enumerate(bounds):
        lo_t = torch.tensor(lo, dtype=raw.dtype, device=raw.device)
        hi_t = torch.tensor(hi, dtype=raw.dtype, device=raw.device)
        out.append(lo_t + (hi_t - lo_t) * torch.sigmoid(raw[..., i]))
    return out


def _fit_single_restart(
    v_np,
    vl_np,
    g_np,
    a_np,
    bounds,
    rest_mask,
    n_epochs,
    lr,
    delta_fixed,
    device,
):
    nv = torch.as_tensor(v_np, dtype=torch.float32, device=device)
    nl = torch.as_tensor(vl_np, dtype=torch.float32, device=device)
    ng = torch.as_tensor(g_np, dtype=torch.float32, device=device)
    na = torch.as_tensor(a_np, dtype=torch.float32, device=device)
    nm = torch.as_tensor(rest_mask.astype(np.float32), dtype=torch.float32, device=device)

    n_param_raw = len(bounds)
    raw = torch.nn.Parameter(torch.zeros(n_param_raw, device=device))
    with torch.no_grad():
        raw.copy_(torch.randn(n_param_raw, device=device) * 0.35)

    if delta_fixed is not None:
        delta = torch.tensor(delta_fixed, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([raw], lr=float(lr))
    denom = torch.clamp(nm.sum(), min=1.0)
    loss_best = math.inf

    def _forward():
        if delta_fixed is None:
            v0t, s0t, att, bt, Tt, deltat = _box_params(raw.unsqueeze(0), bounds)
        else:
            v0t, s0t, att, bt, Tt = _box_params(raw.unsqueeze(0), bounds)
            deltat = delta
        pred = idm_accel_torch(nv, nl, ng, v0t, s0t, att, bt, Tt, deltat)
        resid = pred - na
        return torch.sum((resid ** 2) * nm) / denom, pred.detach()

    for ep in range(n_epochs):
        optimizer.zero_grad()
        loss, _ = _forward()
        loss.backward()
        optimizer.step()
        lb = loss.item()
        if lb < loss_best:
            loss_best = lb

    with torch.no_grad():
        _, pred_f = _forward()
        rmse = float(torch.sqrt(torch.sum(((pred_f - na) ** 2) * nm) / denom).item())

    with torch.no_grad():
        if delta_fixed is None:
            unpacked = _box_params(raw.unsqueeze(0), bounds)
            v0, s0, a, b, T, dd = [float(u.squeeze().item()) for u in unpacked]
        else:
            unpacked = _box_params(raw.unsqueeze(0), bounds)
            v0, s0, a, b, T = [float(u.squeeze().item()) for u in unpacked]
            dd = float(delta_fixed)
    params = dict(v0=v0, s0=s0, a=a, b=b, T=T, delta=dd)
    return rmse, loss_best, params


def extract_samples_from_segments(
    paths,
    gap_min,
    gap_max,
    vmin,
    vmax_gap_outlier_speed,
    time_column="sim_time_s",
    time_fallback_column="timestamp",
):
    v_list = []
    vl_list = []
    g_list = []
    a_list = []

    for fp in paths:
        rows = []
        with open(fp, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames is None:
                continue
            rows = list(rd)
        if not rows:
            continue
        rows = [hydrate_following_row(dict(r)) for r in rows]
        rows = _sort_rows_by_time(rows, time_column, time_fallback_column)
        for r in rows:
            ev = _row_value_csv(r, "ego_v_long")
            if ev is None:
                ev = _parse_float(r.get("ego_speed"))
            lv = _row_value_csv(r, "lead_v_long")
            if lv is None:
                lv = _parse_float(r.get("lead_speed"))
            gap = _row_value_csv(r, "distance_headway")
            accel = _row_value_csv(r, "ego_a_long")
            if accel is None:
                accel = _parse_float(r.get("ego_acceleration"))
            if ev is None or lv is None or gap is None or accel is None:
                continue
            if (
                abs(float(ev)) > vmax_gap_outlier_speed
                or abs(float(lv)) > vmax_gap_outlier_speed
            ):
                continue
            gv = float(gap)
            if gv < gap_min or gv > gap_max or not math.isfinite(gv):
                continue
            th = _parse_float(r.get("time_headway"))
            if th is not None and abs(th - 999.0) < 1e-3:
                continue
            if abs(float(ev)) < vmin:
                continue
            v_list.append(float(ev))
            vl_list.append(float(lv))
            g_list.append(gv)
            a_list.append(float(accel))

    if not v_list:
        return None
    return (
        np.asarray(v_list, dtype=np.float64),
        np.asarray(vl_list, dtype=np.float64),
        np.asarray(g_list, dtype=np.float64),
        np.asarray(a_list, dtype=np.float64),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_calibrated",
    )
    ap.add_argument(
        "--time_column",
        type=str,
        default="sim_time_s",
        help="Sort rows within each CSV by this column when all values parse (sim grid).",
    )
    ap.add_argument(
        "--time_fallback_column",
        type=str,
        default="timestamp",
        help="Fallback time column for sorting when --time_column has missing entries.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/idm_per_driver",
    )
    ap.add_argument("--drivers", type=str, default="", help="Comma ids; empty = all.")
    ap.add_argument("--gap_min", type=float, default=1.0, help="Min gap (m) to keep.")
    ap.add_argument("--gap_max", type=float, default=300.0, help="Max gap (m) to keep.")
    ap.add_argument(
        "--min_speed",
        type=float,
        default=0.4,
        help="Discard rows below this |ego| speed (reduces unrealistic IDM regimes).",
    )
    ap.add_argument(
        "--max_speed_keep",
        type=float,
        default=45.0,
        help="Drop rows with unrealistic speeds (possible unit errors).",
    )
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--n_restarts", type=int, default=5)
    ap.add_argument(
        "--delta_fixed",
        type=float,
        default=4.0,
        help="If >0 freeze exponent at this delta; if <=0 also fit delta in [2,6].",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    rng_main = random.Random(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    segs = _discover_following_csvs(args.data_dir)
    if not segs:
        print(
            "No driving_data.csv or segment_*.csv under {}; check --data_dir.".format(args.data_dir),
            file=sys.stderr,
        )
        return 2

    by_driver = {}
    for p in segs:
        d = _extract_driver_id(p)
        by_driver.setdefault(d, []).append(p)

    want = _parse_ids(args.drivers)
    drivers = sorted(
        (
            want
            if want
            else sorted(by_driver.keys(), key=lambda d: int(d[1:]) if d.startswith("T") and d[1:].isdigit() else 999)
        ),
        key=lambda d: int(d[1:]) if d.startswith("T") and d[1:].isdigit() else 999,
    )

    bounds_v0 = (1.5, max(42.0, args.max_speed_keep * 2.0))
    bounds_s0 = (0.5, 8.0)
    bounds_a = (0.2, 4.5)
    bounds_b = (0.3, 5.5)
    bounds_T = (0.35, 4.5)
    bounds_delta = (2.0, 6.0)

    if args.delta_fixed and args.delta_fixed > 0:
        fit_bounds = [bounds_v0, bounds_s0, bounds_a, bounds_b, bounds_T]
        delta_fixed = float(args.delta_fixed)
    else:
        fit_bounds = [bounds_v0, bounds_s0, bounds_a, bounds_b, bounds_T, bounds_delta]
        delta_fixed = None

    os.makedirs(args.out_dir, exist_ok=True)
    aggregate = dict(version=1, data_dir=os.path.abspath(args.data_dir), drivers=[])

    summary_rows = []

    for d in drivers:
        paths = by_driver.get(d, [])
        pack = extract_samples_from_segments(
            paths,
            args.gap_min,
            args.gap_max,
            args.min_speed,
            args.max_speed_keep,
            time_column=args.time_column,
            time_fallback_column=args.time_fallback_column,
        )
        if pack is None or len(pack[0]) < 50:
            print(
                "[WARN] {} insufficient samples ({}) skipped.".format(
                    d, len(pack[0]) if pack else 0
                )
            )
            continue
        v_np, vl_np, g_np, a_np = pack
        accel_std = float(np.std(a_np)) if len(a_np) > 3 else 1.0

        median_v = float(np.median(np.abs(v_np)))
        vmax_obs = float(np.percentile(np.abs(v_np), 98))
        v0_prior_lo = bounds_v0[0]
        v0_prior_hi = min(bounds_v0[1], max(vmax_obs * 1.4 + 5.0, median_v + 18.0))
        drv_bounds = list(fit_bounds)
        drv_bounds[0] = (v0_prior_lo, v0_prior_hi)

        n_s = len(v_np)
        n_est = max(2000, min(40000, n_s))
        if n_est < n_s:
            ix = rng_main.sample(range(n_s), k=n_est)
            v_np_sub = v_np[ix]
            vl_np_sub = vl_np[ix]
            g_np_sub = g_np[ix]
            a_np_sub = a_np[ix]
        else:
            v_np_sub, vl_np_sub, g_np_sub, a_np_sub = v_np, vl_np, g_np, a_np

        accel_cap = accel_std * 15.0 + 1e-3
        rest_mask = np.abs(a_np_sub) < accel_cap
        if np.count_nonzero(rest_mask) < max(500, len(rest_mask) // 33):
            rest_mask = np.ones(len(rest_mask), dtype=bool)

        global_best_rmse = math.inf
        global_best_loss = math.inf
        global_best_params = None

        restarts = max(1, args.n_restarts)
        for _r in range(restarts):
            seed_local = rng_main.randint(0, 2**31 - 1)
            random.seed(seed_local)
            torch.manual_seed(seed_local)
            rmse, tl, pt = _fit_single_restart(
                v_np_sub,
                vl_np_sub,
                g_np_sub,
                a_np_sub,
                drv_bounds,
                rest_mask.astype(bool),
                args.epochs,
                args.lr,
                delta_fixed,
                device,
            )
            if rmse < global_best_rmse:
                global_best_rmse = rmse
                global_best_loss = tl
                global_best_params = pt

        if global_best_params is None:
            print("[WARN] {} fit failed.".format(d))
            continue

        entry = dict(
            driver_id=d,
            n_samples=len(v_np),
            n_fit_used=len(v_np_sub),
            rmse_accel_mps2=global_best_rmse,
            loss_mse_approx=global_best_loss,
            parameters=global_best_params,
            fit_bounds_repr=str(drv_bounds),
        )
        aggregate["drivers"].append(entry)
        summary_rows.append(
            {
                "driver_id": d,
                "n_samples": len(v_np),
                "rmse_accel": global_best_rmse,
                "v0": global_best_params["v0"],
                "s0": global_best_params["s0"],
                "a": global_best_params["a"],
                "b": global_best_params["b"],
                "T": global_best_params["T"],
                "delta": global_best_params["delta"],
            }
        )

        drv_dir = os.path.join(args.out_dir, d)
        os.makedirs(drv_dir, exist_ok=True)
        with open(os.path.join(drv_dir, "idm.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, sort_keys=False)
        print(
            "{} n={} rmse_accel={:.4f}  v0={:.2f} s0={:.2f} a={:.2f} b={:.2f} T={:.2f} delta={:.2f}".format(
                d,
                len(v_np),
                global_best_rmse,
                global_best_params["v0"],
                global_best_params["s0"],
                global_best_params["a"],
                global_best_params["b"],
                global_best_params["T"],
                global_best_params["delta"],
            )
        )

    all_path = os.path.join(args.out_dir, "idm_all_drivers.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=False)

    csv_path = os.path.join(args.out_dir, "idm_per_driver_summary.csv")
    if summary_rows:
        keys = summary_rows[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(keys))
            w.writeheader()
            w.writerows(summary_rows)

    print("Wrote {} and {}".format(all_path, csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
