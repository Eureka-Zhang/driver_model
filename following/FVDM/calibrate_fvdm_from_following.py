# -*- coding: utf-8 -*-
"""
Per-driver calibration of the Full Velocity Difference (FVDM) longitudinal law with **fixed**
comfort spacing ``s_c`` (from ``extract_sc_comfort_spacing.py`` outputs).

Dynamics::

    a(t) = κ [ V(s) − v ] + λ Δv,

    V(s) = (v_max / 2) [ tanh((s − s_c)/β) + tanh(s_c/β) ],

with **statistics-first** freezing of ``v_max`` and ``β``, then **two-variable linear least squares**
(``numpy`` only) for ``κ`` and ``λ``. This avoids the previous failure mode where nonlinear solvers
pushed ``v_max`` toward absurd values and ``β`` onto box bounds while ``λ`` stuck at ``0``.

**Fixtures (before regression)**

1. ``v_max`` — **never optimized**: ``quantile(ego_v, q_v) + v_margin``, then clamped by
   ``--v_max_hard_cap_m_s``.
2. ``β`` — **never optimized**: ``((s_c_used − gap_p_pct) / 2)`` with clamps
   ``raw < low_raw → beta_low_substitute``, ``raw > beta_cap_m → beta_cap_m`` (defaults match the
   spec you gave earlier: if raw `< 2` use `5`; if raw `> 30` cap at `30`). Invalid / non-positive
   raw ⇒ ``beta_low_substitute``.
3. ``s_c_used`` — CSV median; optional ``--s_c_fit_cap_m`` clamps pathology (e.g. very large ``s_c``).

**Relative speed**

Δv **= v_lead − v_ego** (longitudinal). Export CSVs store ``relative_v_long`` / ``relative_speed`` as
**ego − lead**; this script **never** uses those raw values as Δv—only ``lead_v_long − ego_v_long``,
or ``−relative`` when a speed component is missing.

**λ floor (simulation-safe prior)**

After unconstrained LS, ``λ ← max(λ, --lambda_min_prior)`` (default ``0.05``), then **κ refit**
analytically holding ``λ`` fixed.

Example::

    python3 following/FVDM/calibrate_fvdm_from_following.py \
      --data_dir following/outputs/following_il_clean_gap04 \
      --sc_csv following/outputs/s_c_per_driver.csv \
      --out_csv following/outputs/fvdm_calibrated_per_driver.csv \
      --out_json following/outputs/fvdm_calibrated_per_driver.json
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import extract_sc_comfort_spacing as esc


def _load_sc_csv(path):
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            did = str(row.get("driver_id", "")).strip()
            sc = esc._parse_float(row.get("s_c_median_m"))
            if not did or sc is None or not math.isfinite(float(sc)):
                continue
            mp[did] = float(sc)
    return mp


def _gather_segment_arrays(fp, opts):
    ts_lst = []
    s_lst = []
    v_lst = []
    dv_lst = []
    a_lst = []

    min_ts = float(opts.min_timestamp)
    gap_lo = float(opts.gap_min)
    gap_hi = float(opts.gap_max)
    v_min = float(opts.min_ego_speed)
    clip_a = float(opts.clip_abs_accel)

    with open(fp, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return {}
        for raw in rd:
            r = esc.hydrate_minimal_longitudinal(dict(raw))

            ts = esc._parse_float(r.get("timestamp"))
            if ts is None or ts < min_ts:
                continue

            s = esc._parse_float(r.get("distance_headway"))
            if s is None or not math.isfinite(s):
                continue
            s = float(s)
            if s < gap_lo or s > gap_hi:
                continue

            ev = esc._parse_float(r.get("ego_v_long"))
            if ev is None:
                ev = esc._parse_float(r.get("ego_speed"))
            if ev is None or abs(float(ev)) < v_min:
                continue

            accel = esc._parse_float(r.get("ego_a_long"))
            if accel is None:
                accel = esc._parse_float(r.get("ego_acceleration"))
            if accel is None:
                continue
            accel = float(accel)

            # Δv = v_lead − v_ego (FVDM). Segment CSV convention: relative_v_long ≈ ego_v − lead_v.
            lv = esc._parse_float(r.get("lead_v_long"))
            if lv is not None and ev is not None:
                dv = float(lv) - float(ev)
            else:
                raw_rel = esc._parse_float(r.get("relative_v_long"))
                if raw_rel is None:
                    raw_rel = esc._parse_float(r.get("relative_speed"))
                if raw_rel is None:
                    continue
                dv = -float(raw_rel)

            if opts.exclude_invalid_thw:
                th = esc._parse_float(r.get("time_headway"))
                if esc._is_sentinel_thw(th):
                    continue

            ts_lst.append(ts)
            s_lst.append(s)
            v_lst.append(float(ev))
            dv_lst.append(dv)
            a_lst.append(accel)

    if len(s_lst) < int(opts.min_samples):
        return {}

    s_arr = np.asarray(s_lst, dtype=np.float64)
    v_arr = np.asarray(v_lst, dtype=np.float64)
    dv_arr = np.asarray(dv_lst, dtype=np.float64)
    a_arr = np.asarray(a_lst, dtype=np.float64)
    ts_arr = np.asarray(ts_lst, dtype=np.float64)

    ok = np.isfinite(s_arr) & np.isfinite(v_arr) & np.isfinite(dv_arr) & np.isfinite(a_arr)
    if clip_a > 0.0:
        ok &= np.abs(a_arr) <= clip_a

    ts_arr = ts_arr[ok]
    s_arr = s_arr[ok]
    v_arr = v_arr[ok]
    dv_arr = dv_arr[ok]
    a_arr = a_arr[ok]

    if len(s_arr) < int(opts.min_samples):
        return {}

    return {"t": ts_arr, "s": s_arr, "v": v_arr, "dv": dv_arr, "a": a_arr}


def _V_fvdm(s, sc, vmax, beta):
    a = (np.asarray(s, dtype=np.float64) - float(sc)) / float(beta)
    b = float(sc) / float(beta)
    return 0.5 * float(vmax) * (np.tanh(a) + np.tanh(b))


def _rms(a, pred):
    d = np.asarray(a, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d)))


def _sc_used_for_fit(sc_csv, opts):
    if opts.s_c_fit_cap_m is not None and float(opts.s_c_fit_cap_m) > 0:
        return float(min(sc_csv, float(opts.s_c_fit_cap_m)))
    return float(sc_csv)


def _v_max_fixed(v_arr, opts):
    """v_max = percentile(ego_speed, q) + margin, then physical cap."""
    vnp = np.asarray(v_arr, dtype=np.float64)
    pct = float(opts.v_quantile_fraction) * 100.0
    vq = float(np.percentile(vnp, pct))
    vm = vq + float(opts.v_margin_after_p95)
    cap = float(opts.v_max_hard_cap_m_s)
    if vm > cap:
        vm = cap
    return vm, vq


def _beta_fixed_from_stats(sc_used, gaps, opts):
    """β = ((s_c − gap_q) / 2) then piecewise clamps (default: raw<2→5, raw>30→30)."""
    pct = float(opts.gap_p_low_percentile)
    gap_q = float(np.percentile(np.asarray(gaps, dtype=np.float64), pct))
    raw = (float(sc_used) - gap_q) / 2.0
    raw_plain = raw
    low_thresh = float(opts.beta_raw_below_use_substitute_if_lt)
    sub = float(opts.beta_substitute_if_raw_small_m)
    cap = float(opts.beta_cap_m)

    if (not math.isfinite(raw)) or raw <= 0.0:
        beta = sub
        note = "raw_nonpositive_substitute"
    elif raw < low_thresh:
        beta = sub
        note = "raw_lt_{}_substitute".format(low_thresh)
    elif raw > cap:
        beta = cap
        note = "raw_gt_cap"
    else:
        beta = raw
        note = "formula"

    return float(np.clip(beta, float(opts.beta_abs_floor_m), cap)), gap_q, raw_plain, note


def _fit_kappa_lambda_linear_ls(s, sc, vmax, beta, v, dv, a, opts):
    """
    Unconstrained LS on [kappa,lambda] with small ridge; then lambda floor + kappa analytic refit.
    """
    x_g = _V_fvdm(s, sc, vmax, beta) - np.asarray(v, dtype=np.float64)
    x_d = np.asarray(dv, dtype=np.float64)
    a_obs = np.asarray(a, dtype=np.float64)

    X = np.column_stack([x_g, x_d])
    ridge = float(opts.linear_ridge)
    eye2 = ridge * np.eye(2, dtype=np.float64)
    try:
        xtx = X.T @ X + eye2
        xta = X.T @ a_obs
        theta = np.linalg.solve(xtx, xta)
        kap_ls, lam_ls = float(theta[0]), float(theta[1])
    except np.linalg.LinAlgError:
        kap_ls, lam_ls = float(opts.kappa_fallback), float(opts.lambda_fallback_constant)

    lam_raw_ls = lam_ls

    lam_m = float(max(lam_ls, float(opts.lambda_min_prior)))
    lam_was_floored = lam_m > lam_ls + 1e-12

    denom = float(np.dot(x_g, x_g))
    eps = float(opts.kappa_denominator_eps)
    if denom < eps:
        kap_f = kap_ls
    else:
        kap_f = float(np.dot(x_g, (a_obs - lam_m * x_d)) / (denom + eps))

    kap_f = float(np.clip(kap_f, float(opts.kappa_floor_fit), float(opts.kappa_ceiling_fit)))
    lam_f = float(np.clip(lam_m, float(opts.lambda_min_prior), float(opts.lambda_ceiling)))

    pred = kap_f * x_g + lam_f * x_d
    return kap_f, lam_f, kap_ls, lam_raw_ls, lam_was_floored, _rms(a_obs, pred)


def main():
    ap = argparse.ArgumentParser(
        description="Fit FVDM κ,λ by linear regression; freeze v_max, β per statistics + s_c."
    )
    ap.add_argument("--data_dir", type=str, default="following/outputs/following_il_clean_gap04")
    ap.add_argument("--sc_csv", type=str, default="following/outputs/s_c_per_driver.csv")
    ap.add_argument("--out_csv", type=str, default="following/outputs/fvdm_calibrated_per_driver.csv")
    ap.add_argument("--out_json", type=str, default="following/outputs/fvdm_calibrated_per_driver.json")
    ap.add_argument("--drivers", type=str, default="", help="Comma-separated T ids; empty=all in sc_csv")

    ap.add_argument("--min_timestamp", type=float, default=15.0)
    ap.add_argument("--gap_min", type=float, default=2.0)
    ap.add_argument("--gap_max", type=float, default=250.0)
    ap.add_argument("--min_ego_speed", type=float, default=1.0)
    ap.add_argument(
        "--clip_abs_accel",
        type=float,
        default=15.0,
        help="<=0 disables |a| filter",
    )
    ap.add_argument("--exclude_invalid_thw", action="store_true")

    # Sample count
    ap.add_argument("--min_samples", type=int, default=200)

    # s_c anomaly (CSV still recorded; regression uses capped sc if set)
    ap.add_argument(
        "--s_c_fit_cap_m",
        type=float,
        default=None,
        nargs="?",
        const=120.0,
        metavar="CAP_M",
        help=(
            "If set (with or without numeric value): min(s_csv, CAP) feeds V(s) and β-stats. "
            "Plain --s_c_fit_cap_m uses default CAP=120."
        ),
    )

    # v_max from data
    ap.add_argument(
        "--v_quantile_fraction",
        type=float,
        default=0.95,
        help="Fraction in [0,1] for np.percentile on ego longitudinal speed; default 0.95",
    )
    ap.add_argument("--v_margin_after_p95", type=float, default=2.0, help="+m/s beyond speed quantile")
    ap.add_argument(
        "--v_max_hard_cap_m_s",
        type=float,
        default=45.0,
        help="Physical ceiling after quantile+margins (~162 km/h at 45)",
    )

    # β from gap p-low and s_c
    ap.add_argument(
        "--gap_p_low_percentile",
        type=float,
        default=5.0,
        help="Lower tail of distance_headway distribution (usually 5 for p05)",
    )
    ap.add_argument("--beta_substitute_if_raw_small_m", type=float, default=5.0)
    ap.add_argument("--beta_raw_below_use_substitute_if_lt", type=float, default=2.0)
    ap.add_argument("--beta_cap_m", type=float, default=30.0)
    ap.add_argument(
        "--beta_abs_floor_m",
        type=float,
        default=1.5,
        help="Technical floor on β below which tanh blows up numerically",
    )

    # Linear LS + clamps
    ap.add_argument("--linear_ridge", type=float, default=1e-8)
    ap.add_argument("--lambda_min_prior", type=float, default=0.05)
    ap.add_argument("--lambda_ceiling", type=float, default=25.0)
    ap.add_argument("--lambda_fallback_constant", type=float, default=0.15)
    ap.add_argument(
        "--kappa_floor_fit",
        type=float,
        default=1e-3,
        help="Clamp κ upward after λ-floor refit (keeps a minimal gap-feedback channel)",
    )
    ap.add_argument("--kappa_ceiling_fit", type=float, default=80.0)
    ap.add_argument("--kappa_fallback", type=float, default=0.1)
    ap.add_argument("--kappa_denominator_eps", type=float, default=1e-9)

    args = ap.parse_args()

    if args.clip_abs_accel <= 0.0:
        args.clip_abs_accel = float("inf")
    # nargs='?' optional: unspecified -> getattr None; CONST only when '--s_c_fit_cap_m' bare

    sc_map = _load_sc_csv(args.sc_csv)
    if not sc_map:
        raise SystemExit("Empty or unreadable --sc_csv: {}".format(args.sc_csv))

    drv_filter = esc.parse_driver_filter(args.drivers)
    segments = esc.discover_segment_csvs(args.data_dir)
    by_dr = {}
    for p in segments:
        did = esc.extract_driver_id(p)
        by_dr.setdefault(did, []).append(p)

    rows_out = []
    json_obj = {}

    cols = [
        "driver_id",
        "s_c_csv_m",
        "s_c_used_m",
        "gap_p_low_m",
        "beta_raw_stat_m",
        "n_pts_used",
        "kappa",
        "lambda_vel_diff",
        "lambda_ls_unconstrained",
        "lambda_floored_yes_no",
        "v_max_m_s",
        "v_abs_at_quantile_m_s",
        "beta_m",
        "rms_accel_residual_m_s2",
        "solver",
        "segments_merged",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)

    sorted_drivers = sorted(
        sc_map.keys(),
        key=lambda d: int(d[1:]) if (d.startswith("T") and d[1:].isdigit()) else 999,
    )

    for did in sorted_drivers:
        if drv_filter is not None and did not in drv_filter:
            continue
        sc_csv = float(sc_map[did])
        sc_used = _sc_used_for_fit(sc_csv, args)
        paths = by_dr.get(did, [])
        if not paths:
            print("[WARN] no segments for {}; skip".format(did))
            continue

        blobs = []
        seg_ok = 0
        for fp in sorted(paths):
            ar = _gather_segment_arrays(fp, args)
            if not ar:
                continue
            blobs.append(ar)
            seg_ok += 1

        if not blobs:
            print("[WARN] {}: no usable points after filters; skip".format(did))
            continue

        s = np.concatenate([b["s"] for b in blobs])
        v = np.concatenate([b["v"] for b in blobs])
        dv = np.concatenate([b["dv"] for b in blobs])
        a_obs = np.concatenate([b["a"] for b in blobs])

        if len(s) < int(args.min_samples):
            print("[WARN] {}: only {} rows (need {}); skip".format(did, len(s), args.min_samples))
            continue

        vmax, vq = _v_max_fixed(v, args)
        beta, gap_q, beta_raw, beta_note = _beta_fixed_from_stats(sc_used, s, args)
        kap, lam, kap_ls, lam_raw, lam_floor_flag, rmse = _fit_kappa_lambda_linear_ls(
            s, sc_used, vmax, beta, v, dv, a_obs, args
        )

        rows_out.append(
            {
                "driver_id": did,
                "s_c_csv_m": "{:.6f}".format(sc_csv),
                "s_c_used_m": "{:.6f}".format(sc_used),
                "gap_p_low_m": "{:.6f}".format(gap_q),
                "beta_raw_stat_m": "{:.6f}".format(beta_raw),
                "n_pts_used": str(len(s)),
                "kappa": "{:.8f}".format(kap),
                "lambda_vel_diff": "{:.8f}".format(lam),
                "lambda_ls_unconstrained": "{:.8f}".format(lam_raw),
                "lambda_floored_yes_no": "yes" if lam_floor_flag else "no",
                "v_max_m_s": "{:.6f}".format(vmax),
                "v_abs_at_quantile_m_s": "{:.6f}".format(vq),
                "beta_m": "{:.6f}".format(beta),
                "rms_accel_residual_m_s2": "{:.6f}".format(rmse),
                "solver": "numpy.linear_ls_ridge_lambda_floor",
                "segments_merged": str(seg_ok),
            }
        )
        json_obj[did] = {
            "s_c_csv_m": sc_csv,
            "s_c_used_for_fit_m": sc_used,
            "s_c_fit_was_capped": sc_used < sc_csv - 1e-6,
            "gap_percentile_used": args.gap_p_low_percentile,
            "gap_at_percentile_m": gap_q,
            "beta_raw_stat_before_substitute_cap_m": beta_raw,
            "beta_formula_note": beta_note,
            "n_pts_used": len(s),
            "kappa_ls_unconstrained": kap_ls,
            "kappa_fit_after_lambda_floor_and_clamp": kap,
            "lambda_ls_unconstrained": lam_raw,
            "lambda_after_min_prior_and_clamp": lam,
            "lambda_min_prior": args.lambda_min_prior,
            "lambda_was_floored": lam_floor_flag,
            "v_max_fixed_m_s": vmax,
            "v_abs_quantile_fraction": args.v_quantile_fraction,
            "v_abs_at_quantile_m_s": vq,
            "v_margin_after_quantile_m_s": args.v_margin_after_p95,
            "beta_fixed_m": beta,
            "rms_accel_residual_m_s2": rmse,
            "solver": "numpy.linear_ls_ridge_lambda_floor",
            "segments_merged": seg_ok,
            "equation": {
                "a": "kappa * ( V(s,s_c_used,v_max_fixed,beta_fixed) - v ) + lambda * delta_v_lead_minus_ego",
                "delta_v": "v_lead_long - v_ego_long (computed from velocities; CSV relative_* stored as ego - lead ⇒ negate if used alone)",
                "frozen": ["v_max", "beta", "s_c_used_for_curve"],
                "fitted": ["kappa", "lambda"],
            },
        }
        print(
            "{}: κ={:.4f} λ={:.4f} (LS_raw={:.4f}) v_max={:.2f} β={:.2f} RMSE_a={:.3f} | n={}".format(
                did,
                kap,
                lam,
                lam_raw,
                vmax,
                beta,
                rmse,
                len(s),
            )
        )

    out_csv = os.path.abspath(args.out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    out_json_path = os.path.abspath(args.out_json)
    try:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump({"drivers": json_obj}, f, ensure_ascii=False, indent=2)
    except IOError as exc:
        print("[WARN] could not write json {}: {}".format(out_json_path, exc))

    print("[OK] wrote {} rows → {}".format(len(rows_out), out_csv))


if __name__ == "__main__":
    main()
