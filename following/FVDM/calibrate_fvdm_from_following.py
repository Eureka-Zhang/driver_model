# -*- coding: utf-8 -*-
"""
Per-driver calibration of the Full Velocity Difference (FVDM) longitudinal law with **fixed**
comfort spacing ``s_c``.

**Integrated ``s_c`` extraction** (formerly ``extract_sc_comfort_spacing.py``): when
``--sc_csv`` does not exist on disk, or when ``--force_extract_sc`` is set, this script first
computes per-driver steady-state median headway and writes ``--sc_csv`` (and optional headway /
tertile outputs). If ``--sc_csv`` already exists and extraction is not forced, ``s_c`` is read from
that file.

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
3. ``s_c_used`` — CSV median from extract phase or precomputed `--sc_csv`; optional ``--s_c_fit_cap_m`` clamps pathology (e.g. very large ``s_c``).

**Relative speed**

Δv **= v_lead − v_ego** (longitudinal). Export CSVs store ``relative_v_long`` / ``relative_speed`` as
**ego − lead**; this script **never** uses those raw values as Δv—only ``lead_v_long − ego_v_long``,
or ``−relative`` when a speed component is missing.

**λ floor (simulation-safe prior)**

After unconstrained LS, ``λ ← max(λ, --lambda_min_prior)`` (default ``0.05``), then **κ refit**
analytically holding ``λ`` fixed.

Example (auto-extract ``s_c`` if ``--sc_csv`` missing)::

    python3 following/FVDM/calibrate_fvdm_from_following.py \\
      --data_dir following/outputs/following_calibrated \\
      --sc_csv following/outputs/s_c_per_driver.csv \\
      --out_csv following/outputs/fvdm_calibrated_per_driver.csv \\
      --out_json following/outputs/fvdm_calibrated_per_driver.json

Row filtering uses ``--time_column`` (default ``sim_time_s`` for calibrated rollouts) with
``--time_fallback_column`` (default ``timestamp``). ``--min_timestamp`` is a floor in **seconds on
that chosen axis** (e.g. drop the first ~15 s of simulation time).

**Standalone extract CLI** remains available via ``following/FVDM/extract_sc_comfort_spacing.py``
(thin wrapper around ``extract_sc_comfort_spacing_cli`` in this module).

python3 following/FVDM/calibrate_fvdm_from_following.py \
  --data_dir following/outputs/following_calibrated \
  --sc_csv following/outputs/s_c_per_driver.csv \
  --out_csv following/outputs/fvdm_calibrated_per_driver.csv \
  --out_json following/outputs/fvdm_calibrated_per_driver.json
  
  --force_extract_sc
  
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

# -----------------------------------------------------------------------------
# CSV / longitudinal hydrate (also used by steady-state ``s_c`` extraction)
# -----------------------------------------------------------------------------


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


def hydrate_minimal_longitudinal(row):
    """Fill ego/lead/long relative columns from common aliases (subset of IDM hydrate)."""
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

    return r


_SENTINEL_999_TOL = 1e-3


def _is_sentinel_thw(th):
    if th is None:
        return True
    return abs(float(th) - 999.0) < _SENTINEL_999_TOL


def _median(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    m = n // 2
    if n % 2:
        return s[m]
    return 0.5 * (s[m - 1] + s[m])


def _mean(vals):
    if not vals:
        return None
    return float(sum(vals)) / float(len(vals))


def _percentile_linear(sorted_vals, p):
    """Linear-interpolated percentile; ``p`` in [0, 100]. ``sorted_vals`` must be sorted."""
    if not sorted_vals:
        return None
    s = sorted_vals
    n = len(s)
    if n == 1:
        return s[0]
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (n - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _driver_rank_key(d):
    s = str(d)
    if s.startswith("T") and s[1:].isdigit():
        return int(s[1:])
    return 9999


def extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def parse_driver_filter(s):
    if not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _discover_following_csvs(data_dir):
    """``segment_<n>.csv`` (IL-clean) or ``driving_data.csv`` (calibrated exports)."""
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if fn == "driving_data.csv" or re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _time_seconds_row(r, primary, fallback):
    t = _parse_float(r.get(primary))
    if t is None and fallback and str(fallback) != str(primary):
        t = _parse_float(r.get(fallback))
    return t


def collect_steady_gaps(fp, args):
    """Yield distance_headway samples for steady-state frames (filters on ``args`` namespace)."""
    dh_list = []
    n_read = 0
    with open(fp, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return dh_list, 0
        for raw in rd:
            n_read += 1
            r = hydrate_minimal_longitudinal(dict(raw))

            ego_v = _parse_float(r.get("ego_v_long"))
            if ego_v is None:
                continue
            if args.min_ego_speed > 0 and abs(float(ego_v)) < args.min_ego_speed:
                continue

            accel = _parse_float(r.get("ego_a_long"))
            if accel is None:
                accel = _parse_float(r.get("ego_acceleration"))
            if accel is None:
                continue
            if abs(float(accel)) >= args.a_max:
                continue

            dv = _parse_float(r.get("relative_v_long"))
            if dv is None:
                dv = _parse_float(r.get("relative_speed"))
            if dv is None:
                lv = _parse_float(r.get("lead_v_long"))
                ev = _parse_float(r.get("ego_v_long"))
                if lv is None or ev is None:
                    continue
                dv = float(lv) - float(ev)
            else:
                dv = float(dv)
            if abs(dv) >= args.dv_max:
                continue

            dh = _parse_float(r.get("distance_headway"))
            if dh is None or not math.isfinite(dh):
                continue
            dh = float(dh)
            if dh < args.gap_min or dh > args.gap_max:
                continue

            if args.exclude_invalid_thw:
                th = _parse_float(r.get("time_headway"))
                if _is_sentinel_thw(th):
                    continue

            dh_list.append(dh)

    return dh_list, n_read


def collect_headways_after_timestamp(fp, args, t_cut_s, time_primary, time_fallback):
    """
    Headway samples with time >= t_cut_s (no steady-state filters).
    Uses ``time_primary`` / ``time_fallback`` like calibration row timing.
    """
    dh_list = []
    n_read = 0
    with open(fp, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return dh_list, 0
        for raw in rd:
            n_read += 1
            r0 = dict(raw)
            ts = _time_seconds_row(r0, time_primary, time_fallback)
            if ts is None:
                ts = _parse_float(r0.get("timestamp"))
            if ts is None or float(ts) < float(t_cut_s):
                continue
            r = hydrate_minimal_longitudinal(r0)
            if args.min_ego_speed > 0:
                ev = _parse_float(r.get("ego_v_long"))
                if ev is None or abs(float(ev)) < args.min_ego_speed:
                    continue
            dh = _parse_float(r.get("distance_headway"))
            if dh is None or not math.isfinite(dh):
                continue
            dh = float(dh)
            if dh < args.gap_min or dh > args.gap_max:
                continue
            if args.exclude_invalid_thw:
                th = _parse_float(r.get("time_headway"))
                if _is_sentinel_thw(th):
                    continue
            dh_list.append(dh)
    return dh_list, n_read


def _headway_quantile_dict(sorted_dh):
    if not sorted_dh:
        return None
    out = {
        "mean_m": _mean(sorted_dh),
        "p05_m": _percentile_linear(sorted_dh, 5),
        "p10_m": _percentile_linear(sorted_dh, 10),
        "p25_m": _percentile_linear(sorted_dh, 25),
        "p50_m": _percentile_linear(sorted_dh, 50),
        "p75_m": _percentile_linear(sorted_dh, 75),
        "p90_m": _percentile_linear(sorted_dh, 90),
        "p95_m": _percentile_linear(sorted_dh, 95),
    }
    return out


def _tertile_boundaries_across_subjects(values):
    vals = []
    for x in values:
        if x is None:
            continue
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(xf):
            continue
        vals.append(xf)
    vals.sort()
    if len(vals) < 2:
        return None, None
    t_lo = _percentile_linear(vals, 100.0 / 3.0)
    t_hi = _percentile_linear(vals, 200.0 / 3.0)
    return t_lo, t_hi


def _tertile_subject_buckets_by_rank(pairs):
    valid = []
    for d, v in pairs:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        valid.append((d, fv))
    valid.sort(key=lambda z: (z[1], _driver_rank_key(z[0])))
    n = len(valid)
    if n == 0:
        return [], [], []
    a = (n + 2) // 3
    b = (2 * n + 2) // 3
    low = sorted([p[0] for p in valid[:a]], key=_driver_rank_key)
    mid = sorted([p[0] for p in valid[a:b]], key=_driver_rank_key)
    high = sorted([p[0] for p in valid[b:]], key=_driver_rank_key)
    return low, mid, high


def _parse_float_row_field(s):
    if s is None or str(s).strip() == "":
        return None
    return _parse_float(s)


def build_between_subject_tertile_table(rows_out, headway_rows, headway_cut_s):
    metrics_out = []

    def push_metric(name, pairs):
        vals = [p[1] for p in pairs]
        t_lo, t_hi = _tertile_boundaries_across_subjects(vals)
        n_fin = 0
        for x in vals:
            if x is None:
                continue
            try:
                if math.isfinite(float(x)):
                    n_fin += 1
            except (TypeError, ValueError):
                pass
        low_ids, mid_ids, high_ids = _tertile_subject_buckets_by_rank(pairs)
        metrics_out.append(
            {
                "metric": name,
                "n_subjects_with_value": n_fin,
                "T33_across_subjects": t_lo,
                "T67_across_subjects": t_hi,
                "subject_ids_low": low_ids,
                "subject_ids_mid": mid_ids,
                "subject_ids_high": high_ids,
            }
        )

    sc_pairs = [
        (r.get("driver_id"), _parse_float_row_field(r.get("s_c_median_m"))) for r in rows_out
    ]
    push_metric("s_c_median_m | steady-state median headway per subject [m]", sc_pairs)

    nsteady_pairs = []
    for r in rows_out:
        d = r.get("driver_id")
        v = r.get("n_steady_frames")
        try:
            vf = float(v) if v is not None and str(v).strip() != "" else None
        except (TypeError, ValueError):
            vf = None
        nsteady_pairs.append((d, vf))
    push_metric("n_steady_frames | steady-state count per subject [frames]", nsteady_pairs)

    hw_only = [r for r in (headway_rows or []) if r.get("driver_id") not in (None, "ALL")]
    ttag = " | timestamp>={:.3f}s".format(headway_cut_s) if headway_cut_s and headway_cut_s > 0 else ""

    if hw_only:
        for key in (
            "headway_mean_m",
            "headway_p05_m",
            "headway_p10_m",
            "headway_p25_m",
            "headway_p50_m",
            "headway_p75_m",
            "headway_p90_m",
            "headway_p95_m",
            "n_headway_frames",
        ):
            pairs_h = [
                (r.get("driver_id"), _parse_float_row_field(r.get(key))) for r in hw_only
            ]
            if any(p[1] is not None for p in pairs_h):
                unit = "[m]" if key != "n_headway_frames" else "[frames]"
                push_metric("{} {}{}".format(key, unit, ttag), pairs_h)

    return metrics_out


def run_extract_sc_comfort_spacing(
    data_dir,
    want_drivers,
    steady_ns,
    time_primary,
    time_fallback,
    headway_cut_s,
    out_sc_csv,
    out_json,
    out_headway_stats_csv,
    out_between_subject_tertiles_csv,
):
    """
    Median steady headway → s_c per driver from ``segment_*.csv`` / ``driving_data.csv`` under ``data_dir``.
    Returns dict driver_id → float ``s_c_median`` (omit drivers without finite s_c).
    """
    paths = _discover_following_csvs(data_dir)
    if not paths:
        print("No CSVs under {}".format(data_dir), file=sys.stderr)
        raise SystemExit(2)

    by_drv = {}
    for p in paths:
        d = extract_driver_id(p)
        by_drv.setdefault(d, []).append(p)

    rows_out = []
    aggregate = {
        "version": 1,
        "data_dir": os.path.abspath(data_dir),
        "filters": {
            "a_max": steady_ns.a_max,
            "dv_max": steady_ns.dv_max,
            "gap_min": steady_ns.gap_min,
            "gap_max": steady_ns.gap_max,
            "min_ego_speed": steady_ns.min_ego_speed,
            "exclude_invalid_thw": steady_ns.exclude_invalid_thw,
        },
        "definition": "s_c = median(distance_headway) on steady-state frames (|a|<a_max, |Δv|<dv_max)",
        "drivers": [],
    }

    driver_keys = sorted(
        by_drv.keys(),
        key=lambda di: (
            int(di[1:]) if di.startswith("T") and di[1:].isdigit() else 9999
        ),
    )

    for d in driver_keys:
        if want_drivers is not None and d not in want_drivers:
            continue
        plist = by_drv[d]
        all_dh = []
        n_tot = 0
        for fp in plist:
            dh_seg, n_read_this = collect_steady_gaps(fp, steady_ns)
            all_dh.extend(dh_seg)
            n_tot += n_read_this

        sc = _median(all_dh)
        n_steady = len(all_dh)
        row = {
            "driver_id": d,
            "n_segments": str(len(plist)),
            "n_rows_total_scanned_estimate": str(n_tot),
            "n_steady_frames": str(n_steady),
            "s_c_median_m": "" if sc is None else "{:.6f}".format(sc),
            "s_c_notes": ""
            if sc is not None
            else "insufficient steady frames after filters",
        }
        rows_out.append(row)
        aggregate["drivers"].append(
            {
                "driver_id": d,
                "n_segments": len(plist),
                "n_steady_frames": n_steady,
                "s_c_median_m": sc,
                "median_available": sc is not None,
            }
        )

    headway_rows = []
    pool_all = []
    t_h = float(headway_cut_s)
    if t_h > 0:
        for d in driver_keys:
            if want_drivers is not None and d not in want_drivers:
                continue
            plist = by_drv[d]
            all_h = []
            n_tot_h = 0
            for fp in plist:
                seg_h, nrh = collect_headways_after_timestamp(
                    fp, steady_ns, t_h, time_primary, time_fallback
                )
                all_h.extend(seg_h)
                n_tot_h += nrh
            pool_all.extend(all_h)
            st = _headway_quantile_dict(sorted(all_h))
            hr = {
                "driver_id": d,
                "timestamp_cutoff_s": "{:.3f}".format(t_h),
                "n_segments": str(len(plist)),
                "n_rows_scanned": str(n_tot_h),
                "n_headway_frames": str(len(all_h)),
            }
            if st is None:
                hr.update(
                    {
                        "headway_mean_m": "",
                        "headway_p05_m": "",
                        "headway_p10_m": "",
                        "headway_p25_m": "",
                        "headway_p50_m": "",
                        "headway_p75_m": "",
                        "headway_p90_m": "",
                        "headway_p95_m": "",
                        "notes": "no valid frames after cutoff",
                    }
                )
            else:
                hr.update(
                    {
                        "headway_mean_m": "{:.6f}".format(st["mean_m"]),
                        "headway_p05_m": "{:.6f}".format(st["p05_m"]),
                        "headway_p10_m": "{:.6f}".format(st["p10_m"]),
                        "headway_p25_m": "{:.6f}".format(st["p25_m"]),
                        "headway_p50_m": "{:.6f}".format(st["p50_m"]),
                        "headway_p75_m": "{:.6f}".format(st["p75_m"]),
                        "headway_p90_m": "{:.6f}".format(st["p90_m"]),
                        "headway_p95_m": "{:.6f}".format(st["p95_m"]),
                        "notes": "",
                    }
                )
            headway_rows.append(hr)

        st_all = _headway_quantile_dict(sorted(pool_all))
        row_all = {
            "driver_id": "ALL",
            "timestamp_cutoff_s": "{:.3f}".format(t_h),
            "n_segments": "",
            "n_rows_scanned": "",
            "n_headway_frames": str(len(pool_all)),
        }
        if st_all is None:
            row_all.update(
                {
                    "headway_mean_m": "",
                    "headway_p05_m": "",
                    "headway_p10_m": "",
                    "headway_p25_m": "",
                    "headway_p50_m": "",
                    "headway_p75_m": "",
                    "headway_p90_m": "",
                    "headway_p95_m": "",
                    "notes": "no pooled frames",
                }
            )
        else:
            row_all.update(
                {
                    "headway_mean_m": "{:.6f}".format(st_all["mean_m"]),
                    "headway_p05_m": "{:.6f}".format(st_all["p05_m"]),
                    "headway_p10_m": "{:.6f}".format(st_all["p10_m"]),
                    "headway_p25_m": "{:.6f}".format(st_all["p25_m"]),
                    "headway_p50_m": "{:.6f}".format(st_all["p50_m"]),
                    "headway_p75_m": "{:.6f}".format(st_all["p75_m"]),
                    "headway_p90_m": "{:.6f}".format(st_all["p90_m"]),
                    "headway_p95_m": "{:.6f}".format(st_all["p95_m"]),
                    "notes": "pooled across subjects (same filters)",
                }
            )
        headway_rows.append(row_all)
        aggregate["headway_after_timestamp_s"] = {
            "cutoff_s": t_h,
            "uses_steady_filters": False,
            "gap_min_max": [steady_ns.gap_min, steady_ns.gap_max],
            "exclude_invalid_thw": steady_ns.exclude_invalid_thw,
            "min_ego_speed": steady_ns.min_ego_speed,
            "per_driver": headway_rows[:-1],
            "pooled_ALL": headway_rows[-1] if headway_rows else None,
        }

    tert_rec = build_between_subject_tertile_table(
        rows_out, headway_rows, float(headway_cut_s)
    )
    aggregate["between_subject_tertiles"] = tert_rec

    def _resolved_headway_path():
        if out_headway_stats_csv.strip():
            return out_headway_stats_csv
        slug = "{:.1f}".format(float(headway_cut_s)).rstrip("0").rstrip(".")
        if slug == "":
            slug = "0"
        return os.path.join(
            os.path.dirname(os.path.abspath(data_dir)),
            "headway_after{}s_per_driver.csv".format(slug),
        )

    if t_h > 0 and headway_rows:
        hpath = _resolved_headway_path()
        os.makedirs(os.path.dirname(hpath) or ".", exist_ok=True)
        hk = list(headway_rows[0].keys())
        with open(hpath, "w", newline="", encoding="utf-8") as hf:
            w = csv.DictWriter(hf, fieldnames=hk)
            w.writeheader()
            w.writerows(headway_rows)
        print("Wrote headway stats", hpath, "rows=", len(headway_rows), "(incl. ALL)")

    out_csv_p = out_sc_csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv_p)) or ".", exist_ok=True)
    if rows_out:
        keys = list(rows_out[0].keys())
        with open(out_csv_p, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)

    def _tertiles_path():
        if out_between_subject_tertiles_csv.strip():
            return out_between_subject_tertiles_csv
        return os.path.join(
            os.path.dirname(os.path.abspath(data_dir)),
            "between_subject_tertiles.csv",
        )

    tercsv = _tertiles_path()
    if tert_rec:
        os.makedirs(os.path.dirname(tercsv) or ".", exist_ok=True)
        note_ok = (
            "T33/T67: linear interp on cross-subject values. "
            "subject_ids_*: rank-tertile roster (sort by metric ascending, ~n/3 subjects per tier)."
        )
        tfields = [
            "metric",
            "n_subjects_with_value",
            "T33_across_subjects",
            "T67_across_subjects",
            "subject_ids_low",
            "subject_ids_mid",
            "subject_ids_high",
            "note_zh",
        ]
        with open(tercsv, "w", newline="", encoding="utf-8") as tf:
            ww = csv.DictWriter(tf, fieldnames=tfields)
            ww.writeheader()
            for tr in tert_rec:

                def _join(xs):
                    return ",".join(str(x) for x in (xs or []))

                nsub = int(tr["n_subjects_with_value"])
                ww.writerow(
                    {
                        "metric": tr["metric"],
                        "n_subjects_with_value": str(nsub),
                        "T33_across_subjects": ""
                        if tr["T33_across_subjects"] is None
                        else "{:.6f}".format(tr["T33_across_subjects"]),
                        "T67_across_subjects": ""
                        if tr["T67_across_subjects"] is None
                        else "{:.6f}".format(tr["T67_across_subjects"]),
                        "subject_ids_low": _join(tr.get("subject_ids_low")),
                        "subject_ids_mid": _join(tr.get("subject_ids_mid")),
                        "subject_ids_high": _join(tr.get("subject_ids_high")),
                        "note_zh": note_ok if nsub >= 2 else "fewer than 2 subjects with finite values; no tertile boundaries",
                    }
                )
        print("Wrote between-subject tertiles", tercsv, "rows=", len(tert_rec))

    if out_json.strip():
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(aggregate, jf, indent=2, sort_keys=False)

    print("Wrote", out_csv_p, "rows=", len(rows_out))
    ok = sum(1 for r in rows_out if r["s_c_median_m"])
    print("Drivers with finite s_c:", ok, "/", len(rows_out))

    sc_map = {}
    for r in rows_out:
        did = str(r.get("driver_id", "")).strip()
        sc = _parse_float(r.get("s_c_median_m"))
        if did and sc is not None and math.isfinite(float(sc)):
            sc_map[did] = float(sc)
    return sc_map


def extract_sc_comfort_spacing_cli(argv=None):
    """CLI matching the former standalone ``extract_sc_comfort_spacing.py`` script."""
    ap = argparse.ArgumentParser(description="Median headway on steady-state following frames → s_c")
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root with segment_*.csv and/or driving_data.csv",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Per-driver summary CSV (default: <parent of data_dir>/s_c_comfort_spacing_per_driver.csv if empty)",
    )
    ap.add_argument("--out_json", type=str, default="", help="Optional JSON blob with metadata + drivers")
    ap.add_argument("--a_max", type=float, default=1, help="Max |ego longitudinal accel| (m/s²)")
    ap.add_argument("--dv_max", type=float, default=2, help="Max |relative longitudinal speed| (m/s)")
    ap.add_argument("--gap_min", type=float, default=1.0, help="Min distance_headway to keep (m)")
    ap.add_argument("--gap_max", type=float, default=300.0, help="Max distance_headway to keep (m)")
    ap.add_argument(
        "--min_ego_speed",
        type=float,
        default=2.0,
        help="Drop rows below this |ego_v_long| (m/s); 0 disables",
    )
    ap.add_argument(
        "--exclude_invalid_thw",
        action="store_true",
        default=True,
        help="Exclude rows with time_headway ~999",
    )
    ap.add_argument(
        "--no_exclude_invalid_thw",
        dest="exclude_invalid_thw",
        action="store_false",
        help="Keep rows even when time_headway is sentinel 999",
    )
    ap.add_argument("--drivers", type=str, default="", help="Comma list e.g. T1,T9; empty = all")
    ap.add_argument(
        "--headway_stats_after_s",
        type=float,
        default=10.0,
        help="If >0, headway mean & quantiles for time>=this (s). Uses --time_column / --time_fallback_column",
    )
    ap.add_argument(
        "--time_column",
        "--time-column",
        type=str,
        default="sim_time_s",
        help="Elapsed time column for headway-stats cutoff (preferred)",
    )
    ap.add_argument(
        "--time_fallback_column",
        "--time-fallback-column",
        type=str,
        default="timestamp",
        help="Fallback time column when primary missing on a row",
    )
    ap.add_argument("--out_headway_stats_csv", type=str, default="", help="Per-driver headway stats CSV path")
    ap.add_argument("--out_between_subject_tertiles_csv", type=str, default="", help="Between-subject tertiles CSV")
    xargs = ap.parse_args(argv)

    want = parse_driver_filter(xargs.drivers)

    steady_ns = SimpleNamespace(
        a_max=xargs.a_max,
        dv_max=xargs.dv_max,
        gap_min=xargs.gap_min,
        gap_max=xargs.gap_max,
        min_ego_speed=xargs.min_ego_speed,
        exclude_invalid_thw=xargs.exclude_invalid_thw,
    )

    out_csv_final = xargs.out_csv.strip()
    if not out_csv_final:
        out_csv_final = os.path.join(
            os.path.dirname(os.path.abspath(xargs.data_dir)),
            "s_c_comfort_spacing_per_driver.csv",
        )

    run_extract_sc_comfort_spacing(
        xargs.data_dir,
        want,
        steady_ns,
        xargs.time_column,
        xargs.time_fallback_column,
        float(xargs.headway_stats_after_s),
        out_csv_final,
        xargs.out_json,
        xargs.out_headway_stats_csv,
        xargs.out_between_subject_tertiles_csv,
    )
    return 0


# -----------------------------------------------------------------------------
# FVDM calibration (uses ``hydrate_minimal_longitudinal`` helpers above)
# -----------------------------------------------------------------------------


def _load_sc_csv(path):
    mp = {}
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            did = str(row.get("driver_id", "")).strip()
            sc = _parse_float(row.get("s_c_median_m"))
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
            r = hydrate_minimal_longitudinal(dict(raw))

            ts = _time_seconds_row(r, opts.time_column, opts.time_fallback_column)
            if ts is None or ts < min_ts:
                continue

            s = _parse_float(r.get("distance_headway"))
            if s is None or not math.isfinite(s):
                continue
            s = float(s)
            if s < gap_lo or s > gap_hi:
                continue

            ev = _parse_float(r.get("ego_v_long"))
            if ev is None:
                ev = _parse_float(r.get("ego_speed"))
            if ev is None or abs(float(ev)) < v_min:
                continue

            accel = _parse_float(r.get("ego_a_long"))
            if accel is None:
                accel = _parse_float(r.get("ego_acceleration"))
            if accel is None:
                continue
            accel = float(accel)

            lv = _parse_float(r.get("lead_v_long"))
            if lv is not None and ev is not None:
                dv = float(lv) - float(ev)
            else:
                raw_rel = _parse_float(r.get("relative_v_long"))
                if raw_rel is None:
                    raw_rel = _parse_float(r.get("relative_speed"))
                if raw_rel is None:
                    continue
                dv = -float(raw_rel)

            if opts.exclude_invalid_thw:
                th = _parse_float(r.get("time_headway"))
                if _is_sentinel_thw(th):
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
    vnp = np.asarray(v_arr, dtype=np.float64)
    pct = float(opts.v_quantile_fraction) * 100.0
    vq = float(np.percentile(vnp, pct))
    vm = vq + float(opts.v_margin_after_p95)
    cap = float(opts.v_max_hard_cap_m_s)
    if vm > cap:
        vm = cap
    return vm, vq


def _beta_fixed_from_stats(sc_used, gaps, opts):
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
        description="Extract s_c (optional) + fit FVDM κ,λ by linear regression; freeze v_max, β per statistics + s_c."
    )
    ap.add_argument("--data_dir", type=str, default="following/outputs/following_calibrated")
    ap.add_argument(
        "--sc_csv",
        type=str,
        default="following/outputs/s_c_per_driver.csv",
        help=" Written after extraction; loaded when present unless --force_extract_sc",
    )
    ap.add_argument("--force_extract_sc", action="store_true", help="Always recompute + overwrite --sc_csv first")
    ap.add_argument("--out_csv", type=str, default="following/outputs/fvdm_calibrated_per_driver.csv")
    ap.add_argument("--out_json", type=str, default="following/outputs/fvdm_calibrated_per_driver.json")
    ap.add_argument("--drivers", type=str, default="", help="Comma-separated T ids; empty=all in sc_csv / extract")

    # Steady-state extract (integrated from former extract_sc_comfort_spacing CLI)
    ap.add_argument("--steady_a_max", type=float, default=1.0, help="(|a|, |Δv|) filters for s_c extraction")
    ap.add_argument("--steady_dv_max", type=float, default=2.0)
    ap.add_argument("--steady_gap_min", type=float, default=1.0)
    ap.add_argument("--steady_gap_max", type=float, default=300.0)
    ap.add_argument(
        "--steady_min_ego_speed",
        type=float,
        default=2.0,
        help="Min |ego_v_long| for steady + headway stats (m/s); 0 disables",
    )
    ap.add_argument(
        "--steady_exclude_invalid_thw",
        action="store_true",
        default=True,
        help="Exclude THW sentinel ~999 in extract phase",
    )
    ap.add_argument(
        "--steady_no_exclude_invalid_thw",
        dest="steady_exclude_invalid_thw",
        action="store_false",
        help="Keep THW 999 frames in extract phase",
    )
    ap.add_argument(
        "--headway_stats_after_s",
        type=float,
        default=10.0,
        help="Extract phase headway quantiles after this time (s); 0 skips",
    )
    ap.add_argument("--out_sc_json", type=str, default="", help="Optional extract-phase aggregate JSON path")
    ap.add_argument("--out_headway_stats_csv", type=str, default="", help="Optional headway-stats CSV override")
    ap.add_argument("--out_between_subject_tertiles_csv", type=str, default="", help="Optional tertiles CSV override")

    ap.add_argument(
        "--time_column",
        "--time-column",
        dest="time_column",
        type=str,
        default="sim_time_s",
        help="Calibration + extract headway-stats time column",
    )
    ap.add_argument(
        "--time_fallback_column",
        "--time-fallback-column",
        dest="time_fallback_column",
        type=str,
        default="timestamp",
        help="Used when primary time is missing on a row",
    )
    ap.add_argument(
        "--min_timestamp",
        type=float,
        default=15.0,
        help="Calibration: drop rows with time < this (seconds)",
    )
    ap.add_argument("--gap_min", type=float, default=2.0, help="Calibration FVDM row filter distance_headway [m]")
    ap.add_argument("--gap_max", type=float, default=250.0)
    ap.add_argument("--min_ego_speed", type=float, default=1.0, help="Calibration min |ego| (m/s)")
    ap.add_argument(
        "--clip_abs_accel",
        type=float,
        default=15.0,
        help="<=0 disables |a| filter",
    )
    ap.add_argument("--exclude_invalid_thw", action="store_true")

    ap.add_argument("--min_samples", type=int, default=200)

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

    drv_filter_cal = parse_driver_filter(args.drivers)

    sc_path_abs = os.path.abspath(args.sc_csv)
    extract_first = args.force_extract_sc or (not os.path.isfile(sc_path_abs))

    if extract_first:
        print(
            "[INFO] Extract-phase: computing s_c under {!r}; writing {!r}".format(
                os.path.abspath(args.data_dir),
                sc_path_abs,
            )
        )
        steady_ns = SimpleNamespace(
            a_max=args.steady_a_max,
            dv_max=args.steady_dv_max,
            gap_min=args.steady_gap_min,
            gap_max=args.steady_gap_max,
            min_ego_speed=args.steady_min_ego_speed,
            exclude_invalid_thw=args.steady_exclude_invalid_thw,
        )
        sc_map = run_extract_sc_comfort_spacing(
            args.data_dir,
            drv_filter_cal,
            steady_ns,
            args.time_column,
            args.time_fallback_column,
            float(args.headway_stats_after_s),
            sc_path_abs,
            args.out_sc_json,
            args.out_headway_stats_csv,
            args.out_between_subject_tertiles_csv,
        )
    else:
        sc_map = _load_sc_csv(sc_path_abs)

    if not sc_map:
        raise SystemExit(
            "No s_c available (empty after extract or unreadable file): {!r}".format(sc_path_abs)
        )

    segments = _discover_following_csvs(args.data_dir)
    by_dr = {}
    for p in segments:
        did = extract_driver_id(p)
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
        if drv_filter_cal is not None and did not in drv_filter_cal:
            continue
        sc_csv_val = float(sc_map[did])
        sc_used = _sc_used_for_fit(sc_csv_val, args)
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
                "s_c_csv_m": "{:.6f}".format(sc_csv_val),
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
            "s_c_csv_m": sc_csv_val,
            "s_c_used_for_fit_m": sc_used,
            "s_c_fit_was_capped": sc_used < sc_csv_val - 1e-6,
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
