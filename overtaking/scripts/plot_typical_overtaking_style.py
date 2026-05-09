#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Typical overtaking style comparison — **phase-wise** panels + **p4_return_fix** data.

Reads trajectories from ``--data_root`` (e.g. ``overtaking_p4_return_fix``) and phase
boundaries from ``--phase_summary`` (``phase_segments_summary.csv`` produced by
``segment_overtaking_phases.py``). Row indices in the summary match **time-sorted** rows
with valid ``timestamp`` and ``ego_pos_y`` (same as the segmenter).

**Panels (2 rows × 3 columns)**

* **P2 变道** (phase_02_lane_change): headway; lateral acceleration (from ``ego_pos_y``).
* **P3 超车** (phase_03_left_overtake): ego / lead longitudinal speed vs **time in P3** (s);
  headway vs time in P3; legend includes **P3 duration** (超车时间).
* **P4 变道** (phase_04_return): headway; lateral acceleration.

P2/P4 use **normalized phase time** τ∈[0,1] for shape comparison; P3 uses **seconds from P3 start**
so absolute overtaking duration is visible.

**Cluster bundle** (same as before): per-exp ``overtaking_style_prototypes.json`` +
``driver_overtaking_style_clusters.csv`` for prototype drivers and ``t_duration`` matching.
Optional ``--drivers_by_style conservative=T18,neutral=T5,aggressive=T10``.

Examples::

  python3 overtaking/scripts/plot_typical_overtaking_style.py \
    --cluster_bundle overtaking/outputs/overtaking_style_clusters_selected \
    --data_root overtaking/outputs/overtaking_p4_return_fix \
    --phase_summary overtaking/outputs/overtaking_phase_segments_selected_p1p3/phase_segments_summary.csv \
    --out_dir overtaking/outputs/pictures/typical_overtaking_by_exp \
    --drivers_by_style conservative=T20,neutral=T5,aggressive=T12

  python3 overtaking/scripts/plot_typical_overtaking_style.py \\
    --prototypes_json overtaking/outputs/overtaking_style_merged/overtaking_style_prototypes.json \\
    --data_root overtaking/outputs/overtaking_p4_return_fix \\
    --phase_summary overtaking/outputs/overtaking_phase_segments_selected_p1p3/phase_segments_summary.csv \\
    --out overtaking/outputs/pictures/typical_overtaking_merged.png
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np

_STYLE_ORDER = ("conservative", "neutral", "aggressive")
_STYLE_COLORS = {
    "conservative": "#2ca02c",
    "neutral": "#1f77b4",
    "aggressive": "#d62728",
}

_PROTO_JSON = "overtaking_style_prototypes.json"
_DRIVERS_CSV = "driver_overtaking_style_clusters.csv"


def _norm_driver_id(s):
    s = str(s or "").strip().upper()
    m = re.match(r"^T(\d+)$", s)
    if m:
        return "T{}".format(int(m.group(1)))
    return s


def _parse_float(v):
    if v is None or v == "":
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _load_prototypes(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _driver_rows(csv_path):
    by_id = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            did = _norm_driver_id(row.get("driver_id", ""))
            if did:
                by_id[did] = row
    return by_id


def discover_cluster_bundles(bundle_root):
    root = os.path.abspath(bundle_root)
    bundles = []
    for ek in ("exp1", "exp2", "exp3"):
        d = os.path.join(root, ek)
        pj = os.path.join(d, _PROTO_JSON)
        cf = os.path.join(d, _DRIVERS_CSV)
        if os.path.isfile(pj) and os.path.isfile(cf):
            bundles.append((ek, pj, cf))
    if bundles:
        return sorted(bundles, key=lambda x: x[0])

    base = os.path.basename(root.rstrip(os.sep))
    pj = os.path.join(root, _PROTO_JSON)
    cf = os.path.join(root, _DRIVERS_CSV)
    if os.path.isfile(pj) and os.path.isfile(cf):
        tag = base if base in ("exp1", "exp2", "exp3") else "_merged"
        return [(tag, pj, cf)]

    raise SystemExit(
        "[ERR] cluster bundle {!r}: expected exp1|exp2|exp3 subdirs with {} and {}, "
        "or those two files in the directory.".format(root, _PROTO_JSON, _DRIVERS_CSV),
    )


def parse_drivers_by_style(spec):
    if not spec or not str(spec).strip():
        return None
    aliases = {
        "conservative": "conservative",
        "cons": "conservative",
        "neutral": "neutral",
        "neu": "neutral",
        "aggressive": "aggressive",
        "agg": "aggressive",
    }
    out = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().lower()
        v = _norm_driver_id(v.strip())
        if k in aliases and v:
            out[aliases[k]] = v
    missing = [s for s in _STYLE_ORDER if s not in out]
    if missing:
        raise SystemExit(
            "[ERR] --drivers_by_style must set all three styles; missing: {}".format(
                ", ".join(missing),
            ),
        )
    return out


def _experiment_tag_from_file(rel):
    rel = rel.replace("\\", "/")
    m = re.search(r"_exp([123])_o", rel)
    if m:
        return "exp" + m.group(1)
    return None


def _driver_from_rel_file(rel):
    rel = rel.replace("\\", "/")
    m = re.match(r"^(T\d+)/", rel)
    if m:
        return _norm_driver_id(m.group(1))
    return None


def load_phase_summary_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_track_rows(csv_path):
    """Same row order as segment_overtaking_phases: sort by timestamp, require ts + ego_pos_y."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    rows_ts = []
    for row in raw:
        ts = _parse_float(row.get("timestamp"))
        y = _parse_float(row.get("ego_pos_y"))
        if ts is None or y is None:
            continue
        rows_ts.append((ts, row))
    rows_ts.sort(key=lambda x: x[0])
    times = np.asarray([x[0] for x in rows_ts], dtype=np.float64)
    rows = [x[1] for x in rows_ts]
    return times, rows


def _central_derivative(times, vals):
    n = len(times)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    if n >= 3:
        for i in range(1, n - 1):
            dt = times[i + 1] - times[i - 1]
            if dt > 1e-9:
                out[i] = (vals[i + 1] - vals[i - 1]) / dt
    if n >= 2:
        dt = times[1] - times[0]
        if dt > 1e-9:
            out[0] = (vals[1] - vals[0]) / dt
        dt = times[-1] - times[-2]
        if dt > 1e-9:
            out[-1] = (vals[-1] - vals[-2]) / dt
    return out


def lateral_accel_from_slice(times, y):
    """Lateral acceleration (m/s²) from ego_pos_y."""
    vy = _central_derivative(times, np.asarray(y, dtype=np.float64))
    ay = _central_derivative(times, vy)
    return ay


def series_headway_speed(rows_slice):
    dh = []
    es = []
    ls = []
    for row in rows_slice:
        g = _parse_float(row.get("distance_headway"))
        ev = _parse_float(row.get("ego_speed"))
        if ev is None:
            ev = _parse_float(row.get("ego_v_long"))
        lv = _parse_float(row.get("lead_speed"))
        if lv is None:
            lv = _parse_float(row.get("lead_v_long"))
        dh.append(g if g is not None else float("nan"))
        es.append(ev if ev is not None else float("nan"))
        ls.append(lv if lv is not None else float("nan"))
    return (
        np.asarray(dh, dtype=np.float64),
        np.asarray(es, dtype=np.float64),
        np.asarray(ls, dtype=np.float64),
    )


def _normalize_tau(t_seg):
    if len(t_seg) < 2:
        return np.asarray([0.0])
    t0, t1 = float(t_seg[0]), float(t_seg[-1])
    span = max(t1 - t0, 1e-9)
    return (t_seg - t0) / span


def _meta_duration_from_summary(srow):
    """Approx. maneuver span after following (clustering-style), for picking a session."""
    te = _parse_float(srow.get("t_end"))
    tf = _parse_float(srow.get("t_end_following"))
    if te is None or tf is None:
        return None
    return float(te) - float(tf)


def pick_summary_row(summary_rows, driver_id, exp_tag, target_t_duration):
    did = _norm_driver_id(driver_id)
    cands = []
    for sr in summary_rows:
        if str(sr.get("status", "")).strip().lower() != "ok":
            continue
        if _driver_from_rel_file(sr.get("file", "")) != did:
            continue
        ex = _experiment_tag_from_file(sr.get("file", ""))
        if exp_tag not in (None, "_merged") and ex != exp_tag:
            continue
        if exp_tag == "_merged" and ex is None:
            continue
        i1 = _parse_float(sr.get("i_end_following"))
        ir = _parse_float(sr.get("i_p2_end"))
        i3 = _parse_float(sr.get("i_end_left_overtake"))
        if i1 is None or ir is None or i3 is None:
            continue
        if int(ir) <= int(i1) or int(i3) <= int(ir):
            continue
        cands.append(sr)
    if not cands:
        return None
    if target_t_duration is None or not math.isfinite(float(target_t_duration)):
        return cands[0]
    tgt = float(target_t_duration)
    best = None
    best_e = float("inf")
    for sr in cands:
        m = _meta_duration_from_summary(sr)
        if m is None:
            continue
        e = abs(m - tgt)
        if e < best_e:
            best_e = e
            best = sr
    return best if best is not None else cands[0]


def _phase_durations(times, i1, ir, i3):
    """Seconds: P2 [i1,ir), P3 [ir,i3), P4 [i3,n)."""
    n = len(times)
    i1, ir, i3 = int(i1), int(ir), int(i3)
    if i1 < 0 or ir > n or i3 > n or ir <= i1 or i3 <= ir:
        return 0.0, 0.0, 0.0
    t_p2 = float(times[ir - 1] - times[i1])
    t_p3 = float(times[i3 - 1] - times[ir])
    t_p4 = float(times[n - 1] - times[i3]) if i3 < n - 1 else 0.0
    return t_p2, t_p3, t_p4


def _style_drivers_from_proto(styles_block):
    out = {}
    for sty in _STYLE_ORDER:
        if sty not in styles_block:
            continue
        info = styles_block[sty]
        d = _norm_driver_id(str(info.get("prototype_driver", "")).strip())
        if d:
            out[sty] = d
    return out


def plot_typical_overtaking_phases_figure(
    *,
    summary_rows,
    data_root,
    driver_rows,
    styles_json_block,
    style_to_driver,
    exp_tag,
    drivers_by_style_mode,
    out_path,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("[ERR] matplotlib required")

    # Match visualize_following_overtaking: CJK-capable font if installed; English titles avoid missing glyphs.
    matplotlib.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 9), dpi=120)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.42], hspace=0.38, wspace=0.32)

    # Row 0: all headway; row 1: P2/P4 lateral accel + P3 speed
    ax_p2_h = fig.add_subplot(gs[0, 0])
    ax_p3_g = fig.add_subplot(gs[0, 1])
    ax_p4_h = fig.add_subplot(gs[0, 2])
    ax_p2_a = fig.add_subplot(gs[1, 0])
    ax_p3_v = fig.add_subplot(gs[1, 1])
    ax_p4_a = fig.add_subplot(gs[1, 2])
    ax_bar = fig.add_subplot(gs[2, :])

    bar_dur = []  # list of (style, driver_id, p2, p3, p4)
    p3_headway_plotted = False

    for sty in _STYLE_ORDER:
        driver_id = style_to_driver.get(sty)
        if not driver_id:
            print("[WARN] no driver for {!r} — skip".format(sty))
            continue

        row_csv = driver_rows.get(driver_id)
        if not row_csv:
            raise SystemExit("[ERR] driver {} not in drivers CSV".format(driver_id))

        target_td = _parse_float(row_csv.get("t_duration"))
        if target_td is None and sty in (styles_json_block or {}):
            target_td = _parse_float(
                (styles_json_block[sty].get("prototype_features") or {}).get("t_duration"),
            )

        srow = pick_summary_row(summary_rows, driver_id, exp_tag, target_td)
        if srow is None:
            raise SystemExit(
                "[ERR] no phase_summary row for driver {} exp {!r} (status=ok, valid indices).".format(
                    driver_id,
                    exp_tag,
                ),
            )

        rel = str(srow.get("file", "")).replace("\\", "/")
        csv_path = os.path.join(os.path.abspath(data_root), rel)
        if not os.path.isfile(csv_path):
            raise SystemExit("[ERR] missing CSV: {}".format(csv_path))

        i1 = int(float(srow["i_end_following"]))
        ir = int(float(srow["i_p2_end"]))
        i3 = int(float(srow["i_end_left_overtake"]))

        times, rows = load_track_rows(csv_path)
        n = len(times)
        if i1 < 0 or ir > n or i3 > n or ir <= i1 or i3 <= ir:
            raise SystemExit(
                "[ERR] bad phase indices for {}: i1={} ir={} i3={} n={}".format(rel, i1, ir, i3, n),
            )

        ys = np.asarray(
            [_parse_float(r.get("ego_pos_y")) or 0.0 for r in rows],
            dtype=np.float64,
        )

        # P2 [i1:ir], P3 [ir:i3], P4 [i3:n]
        sl_p2 = slice(i1, ir)
        sl_p3 = slice(ir, i3)
        sl_p4 = slice(i3, n)

        t2 = times[sl_p2]
        t3 = times[sl_p3]
        t4 = times[sl_p4]
        rows2 = rows[sl_p2]
        rows3 = rows[sl_p3]
        rows4 = rows[sl_p4]

        c = _STYLE_COLORS.get(sty, "#333333")
        label_base = "{} {}".format(sty, driver_id)

        # P2
        if len(t2) >= 2:
            tau2 = _normalize_tau(t2)
            dh2, _, _ = series_headway_speed(rows2)
            ay2 = lateral_accel_from_slice(t2, ys[sl_p2])
            ax_p2_h.plot(tau2, dh2, color=c, lw=2.0, label=label_base)
            ax_p2_a.plot(tau2, ay2, color=c, lw=1.6, label=label_base)

        # P3 — wall-clock from P3 start
        if len(t3) >= 2:
            t3_rel = t3 - t3[0]
            p3_dur = float(t3[-1] - t3[0])
            dh3, es3, ls3 = series_headway_speed(rows3)
            ax_p3_g.plot(t3_rel, dh3, color=c, lw=1.8, label="{} (headway)".format(label_base))
            p3_headway_plotted = True
            ax_p3_v.plot(
                t3_rel,
                es3,
                color=c,
                lw=1.9,
                label="{} | P3={:.2f}s".format(label_base, p3_dur),
            )
            ax_p3_v.plot(t3_rel, ls3, color=c, lw=1.1, ls="--", alpha=0.65)

        # P4
        if len(t4) >= 2:
            tau4 = _normalize_tau(t4)
            dh4, _, _ = series_headway_speed(rows4)
            ay4 = lateral_accel_from_slice(t4, ys[sl_p4])
            ax_p4_h.plot(tau4, dh4, color=c, lw=2.0, label=label_base)
            ax_p4_a.plot(tau4, ay4, color=c, lw=1.6, label=label_base)

        d2, d3, d4 = _phase_durations(times, i1, ir, i3)
        bar_dur.append((sty, driver_id, d2, d3, d4))

        print(
            "[pick] [{:5s}] {:12s} {:8s}  P2={:.2f}s P3={:.2f}s P4={:.2f}s  {}".format(
                exp_tag,
                sty,
                driver_id,
                d2,
                d3,
                d4,
                rel,
            ),
        )

    mode_note = "manual Tn" if drivers_by_style_mode else "prototype json"
    fig.suptitle(
        "Typical overtaking by phase [{}] — {} | data: {}".format(exp_tag, mode_note, data_root),
        fontsize=11,
        y=1.02,
    )

    ax_p2_h.set_title("P2 lane change — headway")
    ax_p2_h.set_ylabel("Headway (m)")
    ax_p2_h.set_xlabel("τ (normalized)")
    ax_p2_h.legend(loc="best", fontsize=7)
    ax_p2_h.grid(alpha=0.25)
    ax_p2_h.set_xlim(0, 1)

    ax_p3_g.set_title("P3 left overtake — headway")
    ax_p3_g.set_ylabel("Headway (m)")
    ax_p3_g.set_xlabel("Time in P3 (s)")
    if p3_headway_plotted:
        ax_p3_g.axhline(
            0.0,
            color="0.35",
            ls="--",
            lw=1.0,
            zorder=2,
            label="y = 0 (headway reference)",
        )
    ax_p3_g.legend(loc="best", fontsize=7)
    ax_p3_g.grid(alpha=0.25)

    ax_p4_h.set_title("P4 return — headway")
    ax_p4_h.set_ylabel("Headway (m)")
    ax_p4_h.set_xlabel("τ (normalized)")
    ax_p4_h.legend(loc="best", fontsize=7)
    ax_p4_h.grid(alpha=0.25)
    ax_p4_h.set_xlim(0, 1)

    ax_p2_a.set_title("P2 lane change — lateral acceleration")
    ax_p2_a.set_ylabel("a_lat (m/s²)")
    ax_p2_a.set_xlabel("τ (normalized)")
    ax_p2_a.grid(alpha=0.25)
    ax_p2_a.set_xlim(0, 1)
    ax_p2_a.set_ylim(-50.0, 50.0)

    ax_p3_v.set_title("P3 left overtake — speed (solid ego, dashed lead)")
    ax_p3_v.set_ylabel("Speed (m/s)")
    ax_p3_v.set_xlabel("Time in P3 (s)")
    ax_p3_v.legend(loc="best", fontsize=7)
    ax_p3_v.grid(alpha=0.25)

    ax_p4_a.set_title("P4 return — lateral acceleration")
    ax_p4_a.set_ylabel("a_lat (m/s²)")
    ax_p4_a.set_xlabel("τ (normalized)")
    ax_p4_a.grid(alpha=0.25)
    ax_p4_a.set_xlim(0, 1)
    ax_p4_a.set_ylim(-50.0, 50.0)

    # Bar: fixed yellow → green per phase (not duration-mapped)
    _BAR_P2, _BAR_P3, _BAR_P4 = "#FFD54F", "#9CCC65", "#2E7D32"
    if bar_dur:
        x = np.arange(len(bar_dur), dtype=np.float64)
        w = 0.24
        p2s = [b[2] for b in bar_dur]
        p3s = [b[3] for b in bar_dur]
        p4s = [b[4] for b in bar_dur]
        for i in range(len(bar_dur)):
            ax_bar.bar(
                x[i] - w,
                p2s[i],
                width=w * 0.9,
                label="P2" if i == 0 else "_nolegend_",
                color=_BAR_P2,
                edgecolor="0.25",
                linewidth=0.5,
            )
            ax_bar.bar(
                x[i],
                p3s[i],
                width=w * 0.9,
                label="P3" if i == 0 else "_nolegend_",
                color=_BAR_P3,
                edgecolor="0.25",
                linewidth=0.5,
            )
            ax_bar.bar(
                x[i] + w,
                p4s[i],
                width=w * 0.9,
                label="P4" if i == 0 else "_nolegend_",
                color=_BAR_P4,
                edgecolor="0.25",
                linewidth=0.5,
            )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(["{}\n{}".format(b[0], b[1]) for b in bar_dur], fontsize=8)
        ax_bar.set_ylabel("Duration (s)")
        ax_bar.set_title("Phase durations (fixed colors: P2 yellow → P3 yellow-green → P4 green)")
        ax_bar.legend(loc="upper right", fontsize=8)
        ax_bar.grid(axis="y", alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("[saved] {}".format(out_path))


def _parse_exps_filter(s):
    if not s or not str(s).strip():
        return None
    return {x.strip().lower() for x in str(s).split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(
        description="Typical overtaking: P2/P3/P4 panels from phase_segments_summary + p4_return_fix CSVs.",
    )
    ap.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing mirrored driving_data.csv (e.g. overtaking_p4_return_fix).",
    )
    ap.add_argument(
        "--phase_summary",
        type=str,
        required=True,
        help="phase_segments_summary.csv (from segment_overtaking_phases on the same sessions).",
    )
    ap.add_argument(
        "--cluster_bundle",
        type=str,
        default="",
        help="Parent of exp1/exp2/exp3 cluster outputs, or merged folder with prototypes + drivers csv.",
    )
    ap.add_argument(
        "--prototypes_json",
        type=str,
        default="",
        help="Single run: overtaking_style_prototypes.json (with --out).",
    )
    ap.add_argument("--drivers_csv", type=str, default="", help="Single run: override drivers CSV path.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Bundle mode: output directory (typical_overtaking_<exp>.png).",
    )
    ap.add_argument("--out", type=str, default="", help="Single mode: output PNG path.")
    ap.add_argument("--exps", type=str, default="exp1,exp2,exp3", help="Subset for --cluster_bundle.")
    ap.add_argument(
        "--drivers_by_style",
        type=str,
        default="",
        help="conservative=T18,neutral=T5,aggressive=T10 (same drivers for every exp).",
    )
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    phase_summary_path = os.path.abspath(args.phase_summary)
    if not os.path.isdir(data_root):
        raise SystemExit("[ERR] not a directory: {}".format(data_root))
    if not os.path.isfile(phase_summary_path):
        raise SystemExit("[ERR] missing {}".format(phase_summary_path))

    summary_rows = load_phase_summary_rows(phase_summary_path)
    if not summary_rows:
        raise SystemExit("[ERR] empty phase summary: {}".format(phase_summary_path))

    manual_map = parse_drivers_by_style(args.drivers_by_style) if args.drivers_by_style.strip() else None
    exp_filter = _parse_exps_filter(args.exps)

    bundle_mode = bool(args.cluster_bundle.strip())
    single_mode = bool(args.prototypes_json.strip())

    if bundle_mode and single_mode:
        raise SystemExit("[ERR] use either --cluster_bundle or --prototypes_json, not both.")
    if not bundle_mode and not single_mode:
        raise SystemExit("[ERR] pass --cluster_bundle or --prototypes_json.")

    if bundle_mode:
        out_dir = args.out_dir.strip()
        if not out_dir:
            out_dir = os.path.join(os.path.abspath(args.cluster_bundle), "pictures", "typical_overtaking_phases")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        bundles = discover_cluster_bundles(args.cluster_bundle)
        if exp_filter is not None:
            bundles = [(t, p, c) for t, p, c in bundles if t.lower() in exp_filter or t == "_merged"]
        if not bundles:
            raise SystemExit("[ERR] no bundles after --exps filter")

        for exp_tag, pj, cf in bundles:
            proto = _load_prototypes(pj)
            styles_block = proto.get("styles") or {}
            drv = _driver_rows(cf)
            if manual_map:
                style_to_driver = dict(manual_map)
            else:
                style_to_driver = _style_drivers_from_proto(styles_block)
                for sty in _STYLE_ORDER:
                    if sty not in style_to_driver:
                        print("[WARN] [{!r}] missing prototype for {!r}".format(exp_tag, sty))

            out_png = os.path.join(out_dir, "typical_overtaking_{}.png".format(exp_tag))
            plot_typical_overtaking_phases_figure(
                summary_rows=summary_rows,
                data_root=data_root,
                driver_rows=drv,
                styles_json_block=styles_block,
                style_to_driver=style_to_driver,
                exp_tag=exp_tag,
                drivers_by_style_mode=manual_map is not None,
                out_path=out_png,
            )
        print("[OK] bundle mode → {}".format(out_dir))
        return

    proto_path = os.path.abspath(args.prototypes_json)
    drv_csv = (
        os.path.abspath(args.drivers_csv)
        if args.drivers_csv.strip()
        else os.path.join(os.path.dirname(proto_path), _DRIVERS_CSV)
    )
    if not os.path.isfile(proto_path):
        raise SystemExit("[ERR] missing {}".format(proto_path))
    if not os.path.isfile(drv_csv):
        raise SystemExit("[ERR] missing {}".format(drv_csv))

    proto = _load_prototypes(proto_path)
    styles_block = proto.get("styles") or {}
    drv = _driver_rows(drv_csv)
    if manual_map:
        style_to_driver = dict(manual_map)
    else:
        style_to_driver = _style_drivers_from_proto(styles_block)

    out = args.out.strip()
    if not out:
        out = str(Path(proto_path).parent / "pictures" / "typical_overtaking_phases_merged.png")

    plot_typical_overtaking_phases_figure(
        summary_rows=summary_rows,
        data_root=data_root,
        driver_rows=drv,
        styles_json_block=styles_block,
        style_to_driver=style_to_driver,
        exp_tag="_merged",
        drivers_by_style_mode=manual_map is not None,
        out_path=out,
    )


if __name__ == "__main__":
    main()
