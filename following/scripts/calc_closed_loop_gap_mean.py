#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-driver mean following distance on closed-loop outputs.

Default input directory:
    following/outputs/residual_gru_takeover_20s

Usage:
    python3 following/scripts/calc_closed_loop_gap_mean.py

    python3 following/scripts/calc_closed_loop_gap_mean.py \
        --input_dir following/outputs/residual_gru_takeover_20s \
        --min_time_s 20.0 \
        --output_csv following/outputs/residual_gru_takeover_20s/follow_distance_mean_per_driver.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_driver_dirs(input_dir: Path) -> Iterable[Path]:
    dirs: List[Path] = [p for p in input_dir.iterdir() if p.is_dir() and p.name.startswith("T")]

    def _key(p: Path) -> Tuple[int, str]:
        suffix = p.name[1:]
        return (int(suffix), p.name) if suffix.isdigit() else (10**9, p.name)

    return sorted(dirs, key=_key)


def _mean_gap(csv_path: Path, min_time_s: Optional[float]) -> Tuple[int, float]:
    n = 0
    total = 0.0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gap = _safe_float(row.get("distance_headway"))
            if gap is None:
                continue
            if min_time_s is not None:
                t = _safe_float(row.get("sim_time_s"))
                if t is None or t < min_time_s:
                    continue
            n += 1
            total += gap
    if n == 0:
        return 0, float("nan")
    return n, total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mean closed-loop following distance per driver.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("following/outputs/residual_gru_takeover_20s"),
        help="Directory containing per-driver closed-loop outputs (T*/driving_data.csv).",
    )
    parser.add_argument(
        "--min_time_s",
        type=float,
        default=20.0,
        help="Only use rows with sim_time_s >= min_time_s. Set negative to disable time filter.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("following/outputs/residual_gru_takeover_20s/follow_distance_mean_per_driver.csv"),
        help="Output summary CSV path.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    min_time_s = None if args.min_time_s < 0 else float(args.min_time_s)

    if not input_dir.is_dir():
        raise SystemExit("Input directory not found: {}".format(input_dir))

    rows = []
    for driver_dir in _iter_driver_dirs(input_dir):
        csv_path = driver_dir / "driving_data.csv"
        if not csv_path.is_file():
            continue
        n, mean_gap = _mean_gap(csv_path, min_time_s=min_time_s)
        rows.append((driver_dir.name, n, mean_gap))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["driver_id", "n_rows_used", "mean_distance_headway_m", "min_time_s"])
        for driver_id, n, mean_gap in rows:
            writer.writerow(
                [
                    driver_id,
                    n,
                    "{:.6f}".format(mean_gap) if mean_gap == mean_gap else "",
                    "" if min_time_s is None else "{:.3f}".format(min_time_s),
                ]
            )

    print("Saved: {}".format(output_csv))
    print("Per-driver mean distance_headway:")
    for driver_id, n, mean_gap in rows:
        if mean_gap == mean_gap:
            print("  {}: mean={:.3f} m (n={})".format(driver_id, mean_gap, n))
        else:
            print("  {}: mean=NaN (n=0)".format(driver_id))

    valid = [m for _, _, m in rows if m == m]
    if valid:
        overall = sum(valid) / len(valid)
        print("Overall mean of driver means: {:.3f} m".format(overall))


if __name__ == "__main__":
    main()
