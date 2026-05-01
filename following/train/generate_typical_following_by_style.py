# -*- coding: utf-8 -*-
"""
Generate *typical* longitudinal trajectories for each driving style on a fixed scenario.

- Longitudinal: BC-GRU trained on **all drivers** in that style (see train_bc_following_by_style.py).
- Scenario: fixed lead / world columns from ``--common_case_dir`` (default: T12 session tree).
- Lateral jitter: residual pool from the **style prototype** driver (see
  following_style_prototypes.json from cluster_following_style.py).
- Use ``--single_model_dir`` to force the **same** BC-GRU checkpoint for every style
  (e.g. T12 per-driver folder) while keeping prototype-based lateral replay per style.

Example:
  python3 scripts/generate_typical_following_by_style.py \\
    --prototypes_json outputs/following_style_clusters/following_style_prototypes.json \\
    --model_root outputs/il_bc_gru_by_style \\
    --lateral_pool_root outputs/following_il_clean_gap04 \\
    --common_case_dir data/T12/行车/20260421_120610_198_exp1_f \\
    --out_root outputs/typical_following_by_style
"""
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys


STYLES = ("conservative", "neutral", "aggressive")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prototypes_json",
        type=str,
        default=os.path.join(root, "outputs/following_style_clusters/following_style_prototypes.json"),
    )
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
        help="If set (e.g. .../il_bc_gru_per_driver/T12_longitudinal_framewin), use this checkpoint for every style; lateral still uses each style's prototype driver.",
    )
    ap.add_argument(
        "--generate_py",
        type=str,
        default=os.path.join(root, "train/generate_no_driver_following_outputs.py"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_frames", type=int, default=20)
    ap.add_argument("--lane_center_y", type=float, default=-7.625)
    args = ap.parse_args()

    with open(args.prototypes_json, "r", encoding="utf-8") as f:
        pack = json.load(f)
    styles = pack.get("styles") or pack

    os.makedirs(args.out_root, exist_ok=True)

    for sty in STYLES:
        info = styles.get(sty)
        if not info:
            print("[SKIP] no entry for style:", sty)
            continue
        proto = info.get("prototype_driver") or ""
        if not proto:
            print("[SKIP] style={} missing prototype_driver".format(sty))
            continue
        if args.single_model_dir.strip():
            model_dir = args.single_model_dir.strip()
        else:
            model_dir = os.path.join(args.model_root, sty)
        if not os.path.isfile(os.path.join(model_dir, "best_model.pt")):
            print("[SKIP] style={} missing weights under {}".format(sty, model_dir))
            continue
        out_dir = os.path.join(args.out_root, sty)
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
            "original_jitter",
            "--lane_center_y",
            str(args.lane_center_y),
            "--seed",
            str(args.seed),
            "--warmup_frames",
            str(args.warmup_frames),
            "--lateral_pool_data_dir",
            args.lateral_pool_root,
            "--lateral_pool_driver",
            proto,
        ]
        print("[RUN] style={} prototype={} -> {}".format(sty, proto, out_dir))
        subprocess.check_call(cmd)
        print("[OK]", out_dir)

    print("[OK] all done. Outputs under:", args.out_root)


if __name__ == "__main__":
    main()
