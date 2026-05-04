# -*- coding: utf-8 -*-
"""
Generate *typical* longitudinal trajectories for each driving style on a fixed scenario.

- Longitudinal rollout uses ``generate_no_driver_following_outputs.py`` (shared
  ``bc_gru_features`` with ``train_bc_gru.py``, **closed-loop** GRU + kinematic headway).

- Longitudinal: BC-GRU checkpoint under ``model_root/<style>/`` (or ``--single_model_dir``).
- Scenario: fixed lead / world from ``--common_case_dir``.
- Lateral: **merged + moving-average jitter** via ``generate_no_driver_following_outputs.py``
  ``--lateral_mode pooled_mean_smooth`` and ``--lateral_pool_drivers`` (same drivers you used
  in ``train_bc_following_by_style.py`` for that style).

Example::
python3 following/train/generate_typical_following_by_style.py \
  --conservative T9,T16 --neutral T1,T3,T7,T15,T20 --aggressive T4,T6,T11,T12 \
  --model_root following/outputs/il_bc_gru_by_style \
  --lateral_pool_root following/outputs/following_il_clean_gap04 \
  --common_case_dir data/T12/行车/20260421_120610_198_exp1_f \
  --out_root following/outputs/typical_following_by_style
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys


STYLES = ("conservative", "neutral", "aggressive")


def _sort_driver_ids(ids):
    def keyf(d):
        d = d.strip()
        if d.startswith("T") and d[1:].isdigit():
            return int(d[1:])
        return 9999

    return sorted(ids, key=keyf)


def _parse_driver_list(s):
    """Split comma-separated driver ids; dedupe; sort T* numerically."""
    if not s:
        return []
    parts = [x.strip() for x in s.split(",") if x.strip()]
    seen = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return _sort_driver_ids(seen)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(
        description="Typical trajectories per style: same driver lists as style BC training; pooled smoothed lateral.",
    )
    ap.add_argument(
        "--conservative",
        type=str,
        default="",
        help="Comma-separated drivers (lateral pool + must match trained style pool).",
    )
    ap.add_argument("--neutral", type=str, default="")
    ap.add_argument("--aggressive", type=str, default="")
    ap.add_argument(
        "--model_root",
        type=str,
        default=os.path.join(root, "outputs/il_bc_gru_by_style"),
    )
    ap.add_argument(
        "--lateral_pool_root",
        type=str,
        default=os.path.join(root, "outputs/following_il_clean_gap04"),
    )
    ap.add_argument(
        "--common_case_dir",
        type=str,
        default=os.path.join(root, "data/T12/行车/20260421_120610_198_exp1_f"),
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(root, "outputs/typical_following_by_style"),
    )
    ap.add_argument(
        "--single_model_dir",
        type=str,
        default="",
        help="If set, same BC-GRU for every style; lateral still uses each style's driver list.",
    )
    ap.add_argument(
        "--generate_py",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "generate_no_driver_following_outputs.py"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_frames", type=int, default=20)
    ap.add_argument("--lane_center_y", type=float, default=-7.625)
    args = ap.parse_args()

    style_drivers = {
        "conservative": _parse_driver_list(args.conservative),
        "neutral": _parse_driver_list(args.neutral),
        "aggressive": _parse_driver_list(args.aggressive),
    }

    os.makedirs(args.out_root, exist_ok=True)

    for sty in STYLES:
        drivers = style_drivers[sty]
        if len(drivers) < 1:
            print("[SKIP] style={}: empty driver list".format(sty))
            continue
        if args.single_model_dir.strip():
            model_dir = args.single_model_dir.strip()
        else:
            model_dir = os.path.join(args.model_root, sty)
        if not os.path.isfile(os.path.join(model_dir, "best_model.pt")):
            print("[SKIP] style={} missing weights under {}".format(sty, model_dir))
            continue
        out_dir = os.path.join(args.out_root, sty)
        lateral_csv = ",".join(drivers)
        cmd = [
            sys.executable,
            args.generate_py,
            "--data_dir",
            args.common_case_dir,
            "--model_dir",
            model_dir,
            "--out_dir",
            out_dir,
            "--lateral_mode",
            "pooled_mean_smooth",
            "--lane_center_y",
            str(args.lane_center_y),
            "--seed",
            str(args.seed),
            "--warmup_frames",
            str(args.warmup_frames),
            "--lateral_pool_data_dir",
            args.lateral_pool_root,
            "--lateral_pool_drivers",
            lateral_csv,
        ]
        print("[RUN] style={} lateral_pool_drivers={} -> {}".format(sty, lateral_csv, out_dir))
        subprocess.check_call(cmd)
        print("[OK]", out_dir)

    print("[OK] all done. Outputs under:", args.out_root)


if __name__ == "__main__":
    main()
