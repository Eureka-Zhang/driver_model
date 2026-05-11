#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select the lowest-variance following-distance session for each driver.

The picture directory mirrors the calibrated CSV directory, but the actual
``distance_headway`` values live in ``driving_data.csv`` files. If ``--input_dir``
points at ``following/outputs/pictures/following_calibrated``, this script maps
it back to ``following/outputs/following_calibrated`` automatically.

Usage:
    python3 following/scripts/select_min_headway_variance.py

    python3 following/scripts/select_min_headway_variance.py \
        --input_dir following/outputs/pictures/residual_gru_takeover_20s \
        --output_csv following/outputs/residual_gru_takeover_20s/headway_variance_all_sessions.csv 
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        x = float(text)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _driver_key(driver_id: str) -> Tuple[int, str]:
    suffix = driver_id[1:] if driver_id.startswith("T") else driver_id
    return (int(suffix), driver_id) if suffix.isdigit() else (10**9, driver_id)


def _resolve_csv_root(input_dir: Path) -> Path:
    """Map a mirrored picture root to the corresponding calibrated CSV root."""
    p = input_dir.expanduser().resolve()
    parts = list(p.parts)
    try:
        i = parts.index("pictures")
    except ValueError:
        return p
    mapped = Path(*parts[:i], *parts[i + 1 :])
    return mapped if mapped.is_dir() else p


def _iter_driving_csvs(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("driving_data.csv"))


def _driver_from_path(csv_path: Path, root: Path) -> Optional[str]:
    try:
        rel = csv_path.relative_to(root)
    except ValueError:
        rel = csv_path
    for part in rel.parts:
        if part.startswith("T") and part[1:].isdigit():
            return part
    return None


def _session_label(csv_path: Path, root: Path) -> str:
    try:
        rel = csv_path.relative_to(root)
    except ValueError:
        rel = csv_path
    return str(rel.parent).replace("\\", "/")


def _headway_stats(csv_path: Path, min_time_s: Optional[float]) -> Dict[str, object]:
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if min_time_s is not None:
                t = _safe_float(row.get("sim_time_s"))
                if t is None:
                    t = _safe_float(row.get("timestamp"))
                if t is None or t < min_time_s:
                    continue
            gap = _safe_float(row.get("distance_headway"))
            if gap is not None:
                values.append(gap)

    n = len(values)
    if n == 0:
        return dict(n_rows_used=0, mean=float("nan"), variance=float("nan"), std=float("nan"),
                    min_gap=float("nan"), max_gap=float("nan"))

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return dict(
        n_rows_used=n,
        mean=mean,
        variance=variance,
        std=math.sqrt(variance),
        min_gap=min(values),
        max_gap=max(values),
    )


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return "{:.6f}".format(x) if math.isfinite(x) else ""
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute distance_headway variance and select the minimum-variance session per driver.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("following/outputs/following_calibrated"),
        help="Root containing calibrated driving_data.csv files, or the mirrored pictures root.",
    )
    parser.add_argument(
        "--min_time_s",
        type=float,
        default=-1.0,
        help="Only use rows with time >= min_time_s. Default negative value disables filtering.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("following/outputs/following_calibrated/headway_variance_all_sessions.csv"),
        help="CSV listing every session's following-distance variance.",
    )
    parser.add_argument(
        "--selected_csv",
        type=Path,
        default=Path("following/outputs/following_calibrated/headway_variance_selected_min.csv"),
        help="CSV listing the minimum-variance session selected for each driver.",
    )
    args = parser.parse_args()

    input_root = _resolve_csv_root(args.input_dir)
    min_time_s = None if args.min_time_s < 0 else float(args.min_time_s)

    if not input_root.is_dir():
        raise SystemExit("Input directory not found: {}".format(input_root))

    all_rows: List[Dict[str, object]] = []
    by_driver: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for csv_path in _iter_driving_csvs(input_root):
        driver_id = _driver_from_path(csv_path, input_root)
        if driver_id is None:
            continue
        stats = _headway_stats(csv_path, min_time_s=min_time_s)
        row: Dict[str, object] = dict(
            driver_id=driver_id,
            session=_session_label(csv_path, input_root),
            csv_path=str(csv_path),
            min_time_s="" if min_time_s is None else min_time_s,
            **stats,
        )
        all_rows.append(row)
        by_driver[driver_id].append(row)

    if not all_rows:
        raise SystemExit("No driving_data.csv files found under {}".format(input_root))

    all_rows.sort(key=lambda r: (_driver_key(str(r["driver_id"])), str(r["session"])))
    selected_rows: List[Dict[str, object]] = []
    for driver_id in sorted(by_driver, key=_driver_key):
        candidates = [r for r in by_driver[driver_id] if math.isfinite(float(r["variance"]))]
        if not candidates:
            continue
        best = min(candidates, key=lambda r: (float(r["variance"]), str(r["session"])))
        selected_rows.append(best)

    headers = [
        "driver_id",
        "session",
        "csv_path",
        "n_rows_used",
        "mean",
        "variance",
        "std",
        "min_gap",
        "max_gap",
        "min_time_s",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({h: _fmt(row.get(h, "")) for h in headers})

    args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.selected_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({h: _fmt(row.get(h, "")) for h in headers})

    print("Input CSV root: {}".format(input_root))
    print("Saved all sessions: {}".format(args.output_csv))
    print("Saved selected sessions: {}".format(args.selected_csv))
    print("\nSelected minimum-variance session per driver:")
    for row in selected_rows:
        print(
            "  {driver}: var={var:.6f}, std={std:.3f} m, mean={mean:.3f} m, session={session}".format(
                driver=row["driver_id"],
                var=float(row["variance"]),
                std=float(row["std"]),
                mean=float(row["mean"]),
                session=row["session"],
            ),
        )


if __name__ == "__main__":
    main()
