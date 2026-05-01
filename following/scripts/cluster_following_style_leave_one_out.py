# -*- coding: utf-8 -*-
"""
Leave-one-driver-out (LODO) stability test for ``cluster_following_style.py``.

For each driver D, exclude D, re-run the same k-means + style naming on the remaining
drivers, and compare each remaining driver's ``style_label`` to the **full-cohort
baseline** (all drivers clustered together).

Outputs (under ``--out_dir``):

- ``loo_pairwise.csv`` — one row per (excluded_driver, remaining_driver)
- ``loo_impact_by_excluded.csv`` — how many labels flipped when each driver is removed
- ``loo_stability_by_driver.csv`` — per driver, how often LOO disagreed with baseline
- ``loo_style_counts_by_driver.csv`` — per driver, how many LOO folds assigned each of the 3 styles
- ``ANALYSIS.md`` — short automated write-up

Requires at least **4** drivers so that each LOO fold still has **>= 3** subjects for k=3.

Example::

  python3 following/scripts/cluster_following_style_leave_one_out.py \\
    --data_dir following/outputs/following_il_clean_gap04 \\
    --out_dir following/outputs/following_style_loo \\
    --seed 42
"""
from __future__ import print_function

import argparse
import csv
import os
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_FOLLOWING_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

import cluster_following_style as cfs  # noqa: E402


def _sort_driver_key(d):
    d = d.strip()
    if d.startswith("T") and d[1:].isdigit():
        return int(d[1:])
    return 9999


def _build_rows(by_driver, drivers):
    rows = []
    for driver in sorted(drivers, key=_sort_driver_key):
        m = cfs._summarize_driver(by_driver[driver])
        m["driver_id"] = driver
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Leave-one-driver-out stability vs full-cohort style clustering."
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(_FOLLOWING_ROOT, "outputs", "following_il_clean_gap04"),
        help="Root containing T*/.../segment_*.csv (same as cluster_following_style.py).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(_FOLLOWING_ROOT, "outputs", "following_style_loo"),
        help="Directory for CSV reports and ANALYSIS.md.",
    )
    ap.add_argument(
        "--cluster_dim_weights",
        type=str,
        default="2,1,1,1,1,1,1,1,1,1,1",
        help="Same as cluster_following_style.py --cluster_dim_weights.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dim_weights = cfs.parse_cluster_dim_weights(args.cluster_dim_weights)

    paths = cfs._discover_segments(args.data_dir)
    by_driver = {}
    for p in paths:
        by_driver.setdefault(cfs._driver_id(p), []).append(p)

    all_drivers = sorted(by_driver.keys(), key=_sort_driver_key)
    if len(all_drivers) < 4:
        raise RuntimeError(
            "LOO needs at least 4 drivers (each fold must have >=3 for k=3); got {}".format(
                len(all_drivers)
            )
        )

    os.makedirs(args.out_dir, exist_ok=True)

    rows_full = _build_rows(by_driver, all_drivers)
    cfs.assign_kmeans_styles(rows_full, dim_weights, args.seed)
    baseline = {r["driver_id"]: r["style_label"] for r in rows_full}

    pairwise = []
    impact_rows = []

    for excl in all_drivers:
        rest = [d for d in all_drivers if d != excl]
        rows_loo = _build_rows(by_driver, rest)
        cfs.assign_kmeans_styles(rows_loo, dim_weights, args.seed)
        loo_map = {r["driver_id"]: r["style_label"] for r in rows_loo}
        changed = []
        for d in rest:
            b_sty = baseline[d]
            l_sty = loo_map[d]
            match = "Yes" if b_sty == l_sty else "No"
            pairwise.append(
                {
                    "excluded_driver": excl,
                    "driver_id": d,
                    "style_baseline": b_sty,
                    "style_loo": l_sty,
                    "match": match,
                }
            )
            if b_sty != l_sty:
                changed.append(d)
        impact_rows.append(
            {
                "excluded_driver": excl,
                "n_remaining": str(len(rest)),
                "n_labels_changed": str(len(changed)),
                "changed_drivers": ";".join(sorted(changed, key=_sort_driver_key)),
            }
        )

    pair_fp = os.path.join(args.out_dir, "loo_pairwise.csv")
    with open(pair_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "excluded_driver",
                "driver_id",
                "style_baseline",
                "style_loo",
                "match",
            ],
        )
        w.writeheader()
        for row in pairwise:
            w.writerow(row)

    impact_fp = os.path.join(args.out_dir, "loo_impact_by_excluded.csv")
    with open(impact_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "excluded_driver",
                "n_remaining",
                "n_labels_changed",
                "changed_drivers",
            ],
        )
        w.writeheader()
        for row in impact_rows:
            w.writerow(row)

    by_d = defaultdict(list)
    for row in pairwise:
        if row["match"] == "No":
            by_d[row["driver_id"]].append(row["excluded_driver"])

    n_folds = len(all_drivers) - 1
    stab_rows = []
    for d in all_drivers:
        mism = by_d.get(d, [])
        stab_rows.append(
            {
                "driver_id": d,
                "style_baseline": baseline[d],
                "n_loo_folds": str(n_folds),
                "n_mismatches_vs_baseline": str(len(mism)),
                "mismatch_rate": "{:.6f}".format(len(mism) / float(n_folds))
                if n_folds
                else "",
                "excluded_when_mismatch": ";".join(sorted(mism, key=_sort_driver_key)),
            }
        )
    stab_fp = os.path.join(args.out_dir, "loo_stability_by_driver.csv")
    with open(stab_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "driver_id",
                "style_baseline",
                "n_loo_folds",
                "n_mismatches_vs_baseline",
                "mismatch_rate",
                "excluded_when_mismatch",
            ],
        )
        w.writeheader()
        for row in stab_rows:
            w.writerow(row)

    loo_style_counts = defaultdict(lambda: defaultdict(int))
    for row in pairwise:
        loo_style_counts[row["driver_id"]][row["style_loo"]] += 1

    count_rows = []
    for d in all_drivers:
        c = loo_style_counts[d]
        n_c = int(c.get("conservative", 0))
        n_n = int(c.get("neutral", 0))
        n_a = int(c.get("aggressive", 0))
        denom = float(n_folds) if n_folds else 1.0
        count_rows.append(
            {
                "driver_id": d,
                "style_baseline": baseline[d],
                "n_loo_folds": str(n_folds),
                "n_conservative": str(n_c),
                "n_neutral": str(n_n),
                "n_aggressive": str(n_a),
                "pct_conservative": "{:.4f}".format(n_c / denom),
                "pct_neutral": "{:.4f}".format(n_n / denom),
                "pct_aggressive": "{:.4f}".format(n_a / denom),
            }
        )
    counts_fp = os.path.join(args.out_dir, "loo_style_counts_by_driver.csv")
    with open(counts_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "driver_id",
                "style_baseline",
                "n_loo_folds",
                "n_conservative",
                "n_neutral",
                "n_aggressive",
                "pct_conservative",
                "pct_neutral",
                "pct_aggressive",
            ],
        )
        w.writeheader()
        for row in count_rows:
            w.writerow(row)

    total_pairs = len(pairwise)
    n_mismatch = sum(1 for r in pairwise if r["match"] == "No")
    rate = (n_mismatch / float(total_pairs)) if total_pairs else 0.0
    top_influential = sorted(
        impact_rows, key=lambda x: -int(x["n_labels_changed"])
    )[: min(10, len(impact_rows))]

    analysis_lines = [
        "# Leave-one-driver-out clustering stability",
        "",
        "## Setup",
        "",
        "- **data_dir**: `{}`".format(args.data_dir),
        "- **seed**: {}".format(args.seed),
        "- **cluster_dim_weights**: `{}` (order: {})".format(
            args.cluster_dim_weights, ", ".join(cfs.CLUSTER_FEATURES)
        ),
        "- **Cohort**: {} drivers (`n={}`)".format(", ".join(all_drivers), len(all_drivers)),
        "",
        "**Baseline**: run `assign_kmeans_styles` on **all** drivers. **LOO**: exclude one driver, re-cluster the rest; compare each remaining driver's label to the baseline.",
        "",
        "## Global summary",
        "",
        "- Pairwise comparisons: **{}** (each is one remaining driver vs baseline for one excluded driver).".format(
            total_pairs
        ),
        "- Label mismatches vs baseline: **{}** ({:.1%} of pairwise rows).".format(
            n_mismatch, rate
        ),
        "",
        "**How to read this**: z-scoring and k-means are fit on whoever is in the cohort. Removing one driver changes everyone else's normalized coordinates and cluster boundaries, so some label flips are expected. A **high** mismatch rate means the three-style assignment is **not stable** under small cohort changes; consider more drivers, softer labels (probabilities), or fixed thresholds instead of global k-means.",
        "",
        "## Most influential exclusions (by number of changed labels)",
        "",
        "| excluded_driver | n_remaining | n_labels_changed |",
        "|-----------------|------------:|-----------------:|",
    ]
    for r in top_influential:
        analysis_lines.append(
            "| {} | {} | {} |".format(
                r["excluded_driver"], r["n_remaining"], r["n_labels_changed"]
            )
        )

    analysis_lines.extend(
        [
            "",
            "## Per-driver LOO style frequencies",
            "",
            (
                "For each driver: across the **{:d}** folds where they remain (each exclusion is "
                "some other driver once), tally how often `style_loo` is conservative / neutral / "
                "aggressive. The three integers always sum to {:d}. **style_baseline** is the "
                "full-cohort assignment."
            ).format(n_folds, n_folds),
            "",
            "| driver_id | style_baseline | n_loo_fold | conservative | neutral | aggressive |",
            "|-----------|----------------|-----------:|---------------:|--------:|-----------:|",
        ]
    )
    count_rows_sorted = sorted(count_rows, key=lambda x: _sort_driver_key(x["driver_id"]))
    for crow in count_rows_sorted:
        analysis_lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                crow["driver_id"],
                crow["style_baseline"],
                crow["n_loo_folds"],
                crow["n_conservative"],
                crow["n_neutral"],
                crow["n_aggressive"],
            )
        )
    analysis_lines.extend(
        [
            "",
            "## Per-driver sensitivity",
            "",
            "Open `loo_stability_by_driver.csv`. **mismatch_rate** = among LOO folds where this driver is still present (excluding each *other* driver once), the fraction of folds where `style_loo != style_baseline`.",
            "",
            "## Output files",
            "",
            "| file | content |",
            "|------|---------|",
            "| `loo_pairwise.csv` | per excluded driver, each remaining driver's baseline vs LOO label |",
            "| `loo_impact_by_excluded.csv` | how many drivers changed label when a given driver is removed |",
            "| `loo_stability_by_driver.csv` | per driver mismatch count and rate |",
            "| `loo_style_counts_by_driver.csv` | per driver: LOO histogram over the three styles (+ baseline + percentages) |",
            "",
        ]
    )

    analysis_fp = os.path.join(args.out_dir, "ANALYSIS.md")
    with open(analysis_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(analysis_lines))

    print("[OK] baseline drivers:", len(all_drivers))
    print("[OK] pairwise:", pair_fp)
    print("[OK] impact:", impact_fp)
    print("[OK] stability:", stab_fp)
    print("[OK] style counts:", counts_fp)
    print("[OK] analysis:", analysis_fp)
    print(
        "[INFO] global pairwise mismatch rate: {:.2%} ({} / {})".format(
            rate, n_mismatch, total_pairs
        )
    )


if __name__ == "__main__":
    main()
