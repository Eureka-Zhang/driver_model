# -*- coding: utf-8 -*-
"""
Extract per-subject comfort following spacing ``s_c`` (locked anchor) from Phase-1 car-following CSVs.

**Definition (steady-state framing)**  
Among frames where the driver is nearly coasting (small longitudinal accel) and nearly matched to the
lead in speed (small relative speed), take the **median** of bumper-to-bumper headway ``distance_headway``.
Median is robust to occasional long gaps from inattention.

**Filters (defaults match the spec you gave)**  
- ``|a_long| < a_max``   (default 0.2 m/s²): prefer ``ego_a_long``, else ``ego_acceleration``.
- ``|Δv| < dv_max``      (default 2.5 m/s): longitudinal relative speed magnitude; prefers
  ``relative_v_long`` / ``relative_speed`` else ``lead_v_long - ego_v_long`` (aliases hydrated).
- Extra guards (recommended): sane gap range, exclude invalid ``time_headway`` sentinels, optional
  minimum ego speed so rest/parking rows do not dominate.

**Input**: recursively all ``segment_*.csv`` under ``--data_dir`` (e.g. ``following_il_clean_gap04``).

**Output**: CSV (+ optional JSON) with one row per driver id parsed from ``.../Tn/...`` in the path.

**Between-subject tertiles (thirds)**  
For each scalar summary per driver (``s_c_median_m``, ``n_steady_frames``, and headway-derived columns),
compute **cross-subject** tertile boundaries: ``T33`` (~33.3rd pct) and ``T67`` (~66.7th pct) among drivers,
and list **driver ids** in **low / mid / high** buckets (rank split: sort by metric, then ~equal counts
``ceil(n/3)`` per third). ``T33``/``T67`` are distribution cutoffs; tier membership uses the rank split for
unambiguous rostering. Written via ``--out_between_subject_tertiles_csv``.

Example::

  python3 following/FVDM/extract_sc_comfort_spacing.py \\
    --data_dir following/outputs/following_il_clean_gap04 \\
    --out_csv following/outputs/s_c_per_driver.csv \\
    --out_headway_stats_csv following/outputs/headway_after10s_per_driver.csv \\
    --out_between_subject_tertiles_csv following/outputs/between_subject_tertiles.csv
  
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
import sys


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
    """Fill ego/lead/long relative columns from common aliases (subset of fit_idm hydrate)."""
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


def discover_segment_csvs(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if not re.match(r"segment_\d+\.csv$", fn):
                continue
            out.append(os.path.join(root, fn))
    return sorted(out)


def extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def parse_driver_filter(s):
    if not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def collect_steady_gaps(fp, args):
    """
    Yield (distance_headway,) for rows passing steady filters.
    """
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


def _headway_quantile_dict(sorted_dh):
    """sorted_dh: sorted list of headways (m)."""
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
    """
    Tertile cut points on a **cross-subject** sample (one scalar per subject).
    Returns (T_low, T_high): ~33.33rd and ~66.67th percentiles (linear interpolation).
    Values below ``T_low`` ≈ lowest third of subjects; above ``T_high`` ≈ top third.
    """
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


def _parse_float_row_field(s):
    if s is None or str(s).strip() == "":
        return None
    return _parse_float(s)


def _driver_rank_key(d):
    s = str(d)
    if s.startswith("T") and s[1:].isdigit():
        return int(s[1:])
    return 9999


def _tertile_subject_buckets_by_rank(pairs):
    """
    ``pairs``: (driver_id, scalar) with finite values. Sort by scalar then split into
    three ~equal-count groups: low / mid / high (rank tertiles).
    """
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


def build_between_subject_tertile_table(rows_out, headway_rows, headway_cut_s):
    """
    For each per-subject scalar, compute tertile boundaries **across subjects**
    (~33.3% / ~66.7% linear interpolation on the cross-subject value distribution),
    and list **which driver ids** fall in **low / mid / high** rank-tertile buckets
    (~equal subject counts per bucket, ordered by the metric ascending).
    """
    metrics_out = []

    def push_metric(name, pairs):
        """
        ``pairs``: list of (driver_id, value_or_none).
        """
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
        (r.get("driver_id"), _parse_float_row_field(r.get("s_c_median_m")))
        for r in rows_out
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


def collect_headways_after_timestamp(fp, args, t_cut_s):
    """
    All rows with timestamp >= t_cut_s and valid distance_headway in [gap_min, gap_max].
    Does not apply steady-state (a, Δv) filters.
    """
    dh_list = []
    n_read = 0
    with open(fp, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return dh_list, 0
        for raw in rd:
            n_read += 1
            ts = _parse_float(raw.get("timestamp"))
            if ts is None or float(ts) < float(t_cut_s):
                continue
            r = hydrate_minimal_longitudinal(dict(raw))
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


def main():
    ap = argparse.ArgumentParser(description="Median headway on steady-state following frames → s_c")
    ap.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root with segment_*.csv (e.g. following_il_clean_gap04)",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Write per-driver summary CSV (default: <data_dir>/../s_c_comfort_spacing_per_driver.csv if empty)",
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
        help="Drop rows below this |ego_v_long| (m/s); reduces parking/start-up dominance. Use 0 to disable.",
    )
    ap.add_argument(
        "--exclude_invalid_thw",
        action="store_true",
        default=True,
        help="Exclude rows with time_headway ~999 (invalid / non-following kinematics)",
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
        help="If >0, write headway mean & quantiles for timestamp>=this (s). Use 0 to skip.",
    )
    ap.add_argument(
        "--out_headway_stats_csv",
        type=str,
        default="",
        help="Per-driver + ALL pooled headway stats CSV (default: <parent>/headway_after_{t}s_per_driver.csv)",
    )
    ap.add_argument(
        "--out_between_subject_tertiles_csv",
        type=str,
        default="",
        help=(
            "Tertile cutoffs (approx 33.3% / 66.7%) **across drivers** per summary metric; "
            "empty --> <parent>/between_subject_tertiles.csv"
        ),
    )
    args = ap.parse_args()

    want = parse_driver_filter(args.drivers)

    paths = discover_segment_csvs(args.data_dir)
    if not paths:
        print("No segment_*.csv under {}".format(args.data_dir), file=sys.stderr)
        return 2

    by_drv = {}
    for p in paths:
        d = extract_driver_id(p)
        by_drv.setdefault(d, []).append(p)

    rows_out = []
    aggregate = {
        "version": 1,
        "data_dir": os.path.abspath(args.data_dir),
        "filters": {
            "a_max": args.a_max,
            "dv_max": args.dv_max,
            "gap_min": args.gap_min,
            "gap_max": args.gap_max,
            "min_ego_speed": args.min_ego_speed,
            "exclude_invalid_thw": args.exclude_invalid_thw,
        },
        "definition": "s_c = median(distance_headway) on steady-state frames (|a|<a_max, |Δv|<dv_max)",
        "drivers": [],
    }

    driver_keys = sorted(
        by_drv.keys(),
        key=lambda di: (
            int(di[1:])
            if di.startswith("T") and di[1:].isdigit()
            else 9999
        ),
    )

    for d in driver_keys:
        if want is not None and d not in want:
            continue
        plist = by_drv[d]
        all_dh = []
        n_tot = 0
        for fp in plist:
            dh_seg, n_read_this = collect_steady_gaps(fp, args)
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

    # --- Headway mean & quantiles after timestamp cutoff (no steady-state filter) ---
    headway_rows = []
    pool_all = []
    t_h = float(args.headway_stats_after_s)
    if t_h > 0:
        for d in driver_keys:
            if want is not None and d not in want:
                continue
            plist = by_drv[d]
            all_h = []
            n_tot_h = 0
            for fp in plist:
                seg_h, nrh = collect_headways_after_timestamp(fp, args, t_h)
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
            "gap_min_max": [args.gap_min, args.gap_max],
            "exclude_invalid_thw": args.exclude_invalid_thw,
            "min_ego_speed": args.min_ego_speed,
            "per_driver": headway_rows[:-1],
            "pooled_ALL": headway_rows[-1] if headway_rows else None,
        }

    tert_rec = build_between_subject_tertile_table(
        rows_out, headway_rows, float(args.headway_stats_after_s)
    )
    aggregate["between_subject_tertiles"] = tert_rec

    def _out_headway_csv_path():
        if args.out_headway_stats_csv.strip():
            return args.out_headway_stats_csv
        slug = "{:.1f}".format(float(args.headway_stats_after_s)).rstrip("0").rstrip(".")
        if slug == "":
            slug = "0"
        return os.path.join(
            os.path.dirname(os.path.abspath(args.data_dir)),
            "headway_after{}s_per_driver.csv".format(slug),
        )

    if t_h > 0 and headway_rows:
        hpath = _out_headway_csv_path()
        os.makedirs(os.path.dirname(hpath) or ".", exist_ok=True)
        hk = list(headway_rows[0].keys())
        with open(hpath, "w", newline="", encoding="utf-8") as hf:
            w = csv.DictWriter(hf, fieldnames=hk)
            w.writeheader()
            w.writerows(headway_rows)
        print("Wrote headway stats", hpath, "rows=", len(headway_rows), "(incl. ALL)")

    def _out_csv_path():
        if args.out_csv.strip():
            return args.out_csv
        return os.path.join(
            os.path.dirname(os.path.abspath(args.data_dir)),
            "s_c_comfort_spacing_per_driver.csv",
        )

    out_csv_p = _out_csv_path()
    os.makedirs(os.path.dirname(out_csv_p) or ".", exist_ok=True)
    if rows_out:
        keys = list(rows_out[0].keys())
        with open(out_csv_p, "w", newline="", encoding="utf-8") as wf:
            w = csv.DictWriter(wf, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_out)

    def _out_tertiles_csv_path():
        if args.out_between_subject_tertiles_csv.strip():
            return args.out_between_subject_tertiles_csv
        return os.path.join(
            os.path.dirname(os.path.abspath(args.data_dir)),
            "between_subject_tertiles.csv",
        )

    tercsv = _out_tertiles_csv_path()
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
                nsub = int(tr["n_subjects_with_value"])
                def _join(xs):
                    return ",".join(str(x) for x in (xs or []))
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

    if args.out_json.strip():
        with open(args.out_json, "w", encoding="utf-8") as jf:
            json.dump(aggregate, jf, indent=2, sort_keys=False)

    print("Wrote", out_csv_p, "rows=", len(rows_out))
    ok = sum(1 for r in rows_out if r["s_c_median_m"])
    print("Drivers with finite s_c:", ok, "/", len(rows_out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
