# -*- coding: utf-8 -*-
"""
Train one BC-GRU per following *style* (conservative / neutral / aggressive).

Reads driver → style assignments from ``driver_following_style_clusters.csv`` (from
cluster_following_style.py), pools all segment CSVs for drivers in that style, and
trains with the same hyperparameters as per-driver runs (split_within_driver).

Example:
  python3 scripts/train_bc_following_by_style.py \\
    --cluster_csv outputs/following_style_clusters/driver_following_style_clusters.csv \\
    --data_dir outputs/following_il_clean_gap04 \\
    --out_root outputs/il_bc_gru_by_style
"""
from __future__ import print_function

import argparse
import csv
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


def _drivers_by_style(cluster_csv):
    by_style = {s: [] for s in STYLES}
    with open(cluster_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sty = (row.get("style_label") or "").strip().lower()
            did = (row.get("driver_id") or "").strip()
            if sty in by_style and did:
                by_style[sty].append(did)
    return by_style


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cluster_csv",
        type=str,
        default=os.path.join(root, "outputs/following_style_clusters/driver_following_style_clusters.csv"),
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root, "outputs/following_il_clean_gap04"),
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(root, "outputs/il_bc_gru_by_style"),
    )
    ap.add_argument(
        "--train_bc",
        type=str,
        default=os.path.join(root, "train/train_bc_gru.py"),
    )
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_style = _drivers_by_style(args.cluster_csv)
    os.makedirs(args.out_root, exist_ok=True)

    for sty in STYLES:
        drivers = _sort_driver_ids(list(set(by_style[sty])))
        if len(drivers) < 2:
            print("[SKIP] style={} only {} driver(s); need >=2 for split_within_driver.".format(sty, len(drivers)))
            continue
        ds = ",".join(drivers)
        out_dir = os.path.join(args.out_root, sty)
        cmd = [
            sys.executable,
            args.train_bc,
            "--data_dir",
            args.data_dir,
            "--out_dir",
            out_dir,
            "--train_drivers",
            ds,
            "--val_drivers",
            ds,
            "--test_drivers",
            ds,
            "--split_within_driver",
            "--train_ratio",
            "0.7",
            "--val_ratio",
            "0.15",
            "--test_ratio",
            "0.15",
            "--seq_len",
            str(args.seq_len),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
        ]
        print("[RUN]", sty, "drivers:", ds)
        subprocess.check_call(cmd)
        print("[OK] model dir:", out_dir)


if __name__ == "__main__":
    main()
