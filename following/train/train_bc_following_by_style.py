# -*- coding: utf-8 -*-
"""
Train one BC-GRU per following *style* (conservative / neutral / aggressive).

You **manually assign** which drivers belong to each style via comma-separated
``--conservative`` / ``--neutral`` / ``--aggressive`` lists. For each non-empty
style, all ``segment_*.csv`` under ``--data_dir`` for those drivers are pooled and
``train_bc_gru.py`` is run with ``split_within_driver`` (same as per-driver runs).

Example::

  python3 following/train/train_bc_following_by_style.py \
    --conservative T2,T9,T16 \
    --neutral T7,T10,T20 \
    --aggressive T3,T5,T6,T8,T19 \
    --data_dir following/outputs/following_il_clean_gap04 \
    --out_root following/outputs/il_bc_gru_by_style
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
        description="Train BC-GRU per style using explicit driver lists per style.",
    )
    ap.add_argument(
        "--conservative",
        type=str,
        default="",
        help="Comma-separated driver ids for conservative style (e.g. T1,T2,T3).",
    )
    ap.add_argument(
        "--neutral",
        type=str,
        default="",
        help="Comma-separated driver ids for neutral style.",
    )
    ap.add_argument(
        "--aggressive",
        type=str,
        default="",
        help="Comma-separated driver ids for aggressive style.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root, "outputs", "following_il_clean_gap04"),
    )
    ap.add_argument(
        "--out_root",
        type=str,
        default=os.path.join(root, "outputs", "il_bc_gru_by_style"),
    )
    ap.add_argument(
        "--train_bc",
        type=str,
        default=os.path.join(root, "train", "train_bc_gru.py"),
    )
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_style = {
        "conservative": _parse_driver_list(args.conservative),
        "neutral": _parse_driver_list(args.neutral),
        "aggressive": _parse_driver_list(args.aggressive),
    }

    if not any(by_style[s] for s in STYLES):
        ap.error(
            "Provide at least one non-empty --conservative, --neutral, or --aggressive list."
        )

    driver_to_styles = {}
    for sty in STYLES:
        for d in by_style[sty]:
            driver_to_styles.setdefault(d, []).append(sty)
    dup = {d: st for d, st in driver_to_styles.items() if len(st) > 1}
    if dup:
        for d, st_list in sorted(dup.items()):
            print(
                "[WARN] driver {} appears in multiple styles: {} (each style trains a separate model; data may overlap).".format(
                    d, ", ".join(st_list)
                )
            )

    os.makedirs(args.out_root, exist_ok=True)

    for sty in STYLES:
        drivers = by_style[sty]
        if not drivers:
            print("[SKIP] style={} empty (no --{} given).".format(sty, sty))
            continue
        if len(drivers) < 2:
            print(
                "[SKIP] style={} only {} driver(s) {}; need >=2 for split_within_driver.".format(
                    sty, len(drivers), drivers
                )
            )
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
