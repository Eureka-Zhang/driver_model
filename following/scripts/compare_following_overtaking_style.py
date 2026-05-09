# -*- coding: utf-8 -*-
"""
Compare per-subject car-following style vs overtaking style.

Reads:
  - Following: ``following_style_features.csv`` (columns ``driver``, ``label``)
    or any CSV with those columns.
  - Overtaking: ``driver_overtaking_style_clusters.csv`` (``driver_id``, ``style_label``),
    or a directory containing that file, or ``exp1``/``exp2``/``exp3`` subfolders each with it.

Writes a merged CSV, prints contingency counts, optional confusion-matrix PNG.

Example (merged overtaking, single CSV)::

  python3 following/scripts/compare_following_overtaking_style.py \\
    --following_csv following/outputs/following_style_clusters/following_style_features.csv \\
    --overtaking_csv overtaking/outputs/overtaking_style_merged/driver_overtaking_style_clusters.csv \\
    --out_dir following/outputs/style_comparison_merged \\
    --plot

Example (per-experiment overtaking folders)::

  python3 following/scripts/compare_following_overtaking_style.py \\
    --following_csv following/outputs/following_style_clusters/following_style_features.csv \\
    --overtaking_dir overtaking/outputs/overtaking_style_clusters_selected \\
    --out_dir following/outputs/style_comparison_by_exp \\
    --plot
"""
from __future__ import print_function

import argparse
import csv
import json
import os
import re

_STYLE_ORDER = ("conservative", "neutral", "aggressive")


def _norm_driver(s):
    s = str(s).strip()
    if not s:
        return ""
    s = s.upper()
    m = re.match(r"^T(\d+)$", s)
    if m:
        return "T{}".format(int(m.group(1)))
    return s


def _load_following(path):
    by_d = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if "driver" not in (rdr.fieldnames or []):
            raise SystemExit("[ERR] {}: need column 'driver'".format(path))
        lab_col = "label" if "label" in rdr.fieldnames else None
        if lab_col is None:
            for c in ("style_label", "following_label"):
                if c in rdr.fieldnames:
                    lab_col = c
                    break
        if lab_col is None:
            raise SystemExit("[ERR] {}: need column 'label' (or style_label)".format(path))
        for row in rdr:
            d = _norm_driver(row.get("driver", ""))
            if not d:
                continue
            by_d[d] = str(row.get(lab_col, "") or "").strip()
    return by_d


def _load_overtaking_csv(path):
    by_d = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if "driver_id" not in (rdr.fieldnames or []):
            raise SystemExit("[ERR] {}: need column 'driver_id'".format(path))
        for row in rdr:
            d = _norm_driver(row.get("driver_id", ""))
            if not d:
                continue
            by_d[d] = str(row.get("style_label", "") or "").strip()
    return by_d


def _discover_overtaking_tables(overtaking_dir):
    """Return list of (tag, path_to_csv)."""
    root = os.path.abspath(overtaking_dir)
    out = []
    for ek in ("exp1", "exp2", "exp3"):
        fp = os.path.join(root, ek, "driver_overtaking_style_clusters.csv")
        if os.path.isfile(fp):
            out.append((ek, fp))
    if out:
        return sorted(out, key=lambda x: x[0])
    fp = os.path.join(root, "driver_overtaking_style_clusters.csv")
    if os.path.isfile(fp):
        return [("_merged", fp)]
    raise SystemExit(
        "[ERR] no driver_overtaking_style_clusters.csv under {!r} "
        "(expected file or exp1/exp2/exp3 subfolders).".format(root),
    )


def _contingency(following_by_d, overtaking_by_d, drivers):
    """following x overtaking counts for ordered style labels."""
    mat = {f: {o: 0 for o in _STYLE_ORDER} for f in _STYLE_ORDER}
    other_f = set()
    other_o = set()
    for d in drivers:
        fl = following_by_d.get(d, "")
        ol = overtaking_by_d.get(d, "")
        if fl not in mat:
            other_f.add(fl)
            continue
        if ol not in mat[fl]:
            other_o.add(ol)
            continue
        mat[fl][ol] += 1
    return mat, other_f, other_o


def _agreement_rate(following_by_d, overtaking_by_d, drivers):
    n = 0
    same = 0
    for d in drivers:
        if d not in following_by_d or d not in overtaking_by_d:
            continue
        n += 1
        if following_by_d[d] == overtaking_by_d[d]:
            same += 1
    return (float(same) / n) if n else 0.0, same, n


def _plot_confusion(mat, title, out_png):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed; skip plot.")
        return

    labels = list(_STYLE_ORDER)
    M = np.array([[mat[f][o] for o in labels] for f in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=120)
    im = ax.imshow(M, cmap="Blues", vmin=0)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Overtaking style")
    ax.set_ylabel("Following style")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", color="0.15", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[plot] {}".format(out_png))


def main():
    ap = argparse.ArgumentParser(
        description="Merge following vs overtaking style labels per driver.",
    )
    ap.add_argument(
        "--following_csv",
        type=str,
        required=True,
        help="following_style_features.csv (columns driver, label).",
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--overtaking_csv",
        type=str,
        default="",
        help="driver_overtaking_style_clusters.csv",
    )
    g.add_argument(
        "--overtaking_dir",
        type=str,
        default="",
        help="Directory with overtaking CSV or exp1/exp2/exp3 subfolders.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (default: alongside following_csv).",
    )
    ap.add_argument("--plot", action="store_true", help="Write confusion matrix PNG(s).")
    args = ap.parse_args()

    fol_fp = os.path.abspath(args.following_csv)
    if not os.path.isfile(fol_fp):
        raise SystemExit("[ERR] missing {}".format(fol_fp))

    following_by_d = _load_following(fol_fp)

    if args.overtaking_dir:
        ot_tables = _discover_overtaking_tables(args.overtaking_dir)
    else:
        o_fp = os.path.abspath(args.overtaking_csv)
        if not os.path.isfile(o_fp):
            raise SystemExit("[ERR] missing {}".format(o_fp))
        ot_tables = [("_merged", o_fp)]

    overtaking_maps = [(tag, _load_overtaking_csv(p)) for tag, p in ot_tables]

    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(fol_fp), "following_overtaking_style_comparison")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_drivers = sorted(
        set(following_by_d.keys()) | set().union(*[set(m.keys()) for _, m in overtaking_maps]),
        key=lambda x: (int(x[1:]) if x[1:].isdigit() else 999, x),
    )

    # --- wide CSV ---
    otags = [t for t, _ in overtaking_maps]
    fieldnames = ["driver", "following_label"] + ["overtaking_{}".format(t) for t in otags]
    wide_fp = os.path.join(out_dir, "following_overtaking_style_by_driver.csv")
    with open(wide_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for d in all_drivers:
            row = {"driver": d, "following_label": following_by_d.get(d, "")}
            for t, m in overtaking_maps:
                row["overtaking_{}".format(t)] = m.get(d, "")
            w.writerow(row)
    print("[saved] {}".format(wide_fp))

    summary = {
        "following_csv": fol_fp,
        "overtaking_sources": [{"tag": t, "path": p} for t, p in ot_tables],
        "n_following_drivers": len(following_by_d),
        "drivers_in_union": len(all_drivers),
        "bundles": [],
    }

    for t, o_map in overtaking_maps:
        drivers_both = [d for d in all_drivers if d in following_by_d and d in o_map]
        missing_f = sorted(set(o_map.keys()) - set(following_by_d.keys()))
        missing_o = sorted(set(following_by_d.keys()) - set(o_map.keys()))
        agr_rate, same, n = _agreement_rate(following_by_d, o_map, all_drivers)
        mat, other_f, other_o = _contingency(following_by_d, o_map, drivers_both)
        bundle = {
            "tag": t,
            "n_both": len(drivers_both),
            "agreement_rate_among_labeled_both": agr_rate,
            "agreement_count": same,
            "compared_with_following_count": n,
            "missing_following_only": missing_o,
            "missing_overtaking_only": missing_f,
            "contingency_following_rows": {f: mat[f] for f in _STYLE_ORDER},
            "other_following_labels": sorted(other_f),
            "other_overtaking_labels": sorted(other_o),
        }
        summary["bundles"].append(bundle)

        print("\n=== Overtaking [{}] vs following ===".format(t))
        print(
            "  drivers with both labels: {}  |  exact match (same label): {:.1%} ({}/{})".format(
                len(drivers_both),
                agr_rate,
                same,
                n,
            )
        )
        if missing_o:
            print("  [WARN] no overtaking row for: {}".format(", ".join(missing_o)))
        if missing_f:
            print("  [WARN] no following row for: {}".format(", ".join(missing_f)))
        print("  Contingency (row=following, col=overtaking):")
        hdr = "              " + "".join("{:>14s}".format(c) for c in _STYLE_ORDER)
        print(hdr)
        for f in _STYLE_ORDER:
            line = "  {:12s}".format(f) + "".join(
                "{:>14d}".format(mat[f][o]) for o in _STYLE_ORDER
            )
            print(line)

        if args.plot:
            png = os.path.join(out_dir, "confusion_following_vs_overtaking_{}.png".format(t))
            _plot_confusion(mat, "Following vs overtaking [{}]".format(t), png)

    sum_fp = os.path.join(out_dir, "following_overtaking_style_comparison_summary.json")
    with open(sum_fp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[saved] {}".format(sum_fp))
    print("[OK] done")


if __name__ == "__main__":
    main()
