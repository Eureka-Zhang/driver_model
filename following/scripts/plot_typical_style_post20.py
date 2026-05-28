#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot typical following style comparison after takeover (sim_time_s >= 20s).

Trajectory CSVs and legend text are **independent**:
  --trajectory_drivers  which folders to plot (e.g. T9 for conservative curve)
  --legend              legend strings per style (any text you want)

If only --drivers is given, it sets both trajectory and legend (legacy).

Usage::

    python3 following/scripts/plot_typical_style_post20.py

    # One line (recommended; avoid "\\  " before --legend — shell passes a stray " " arg)
    python3 following/scripts/plot_typical_style_post20.py \\
      --trajectory_drivers conservative=T20,neutral=T15,aggressive=T11 \\
      --legend "conservative=Conservative (T16),neutral=Neutral (T15),aggressive=Aggressive (T19)"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "following" / "outputs" / "residual_gru_takeover_20s"
_DEFAULT_OUT = (
    _ROOT
    / "following"
    / "outputs"
    / "pictures"
    / "residual_gru_takeover_20s"
    / "typical_style_comparison_post20s.png"
)

_STYLE_COLORS = {
    "conservative": "#1f77b4",
    "neutral": "#2ca02c",
    "aggressive": "#d62728",
}
_STYLE_ORDER = ("conservative", "neutral", "aggressive")


def _parse_style_map(s: str, flag: str, *, upper_values: bool = False) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit("[ERR] {}: expected style=value, got {!r}".format(flag, part))
        style, val = part.split("=", 1)
        style = style.strip().lower()
        val = val.strip()
        if style not in _STYLE_COLORS:
            raise SystemExit("[ERR] {}: unknown style {!r}".format(flag, style))
        if style in out:
            raise SystemExit("[ERR] {}: duplicate style {!r}".format(flag, style))
        out[style] = val.upper() if upper_values else val
    return out


def _ordered_styles(style_map: Dict[str, str]) -> List[str]:
    return [s for s in _STYLE_ORDER if s in style_map]


def _load_post20_series(csv_path: Path, t_min: float) -> Dict[str, List[float]]:
    t, gap, v, a = [], [], [], []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tt = float(row["sim_time_s"])
                if tt < t_min:
                    continue
                gg = float(row["distance_headway"])
                vv = float(row.get("ego_v_long") or row.get("ego_speed"))
                aa = float(row.get("ego_a_long") or row.get("ego_acceleration"))
            except (KeyError, TypeError, ValueError):
                continue
            t.append(tt)
            gap.append(gg)
            v.append(vv)
            a.append(aa)
    mean_gap = (sum(gap) / len(gap)) if gap else float("nan")
    return {"t": t, "gap": gap, "v": v, "a": a, "mean_gap": mean_gap}


def main() -> None:
    # "\\  --legend" in shell becomes argv [" ", "--legend", ...] and breaks argparse.
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if str(a).strip()]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=_DEFAULT_DATA)
    ap.add_argument(
        "--drivers",
        type=str,
        default="",
        help="Legacy: style=driver_id for both trajectory and legend title (style + driver).",
    )
    ap.add_argument(
        "--trajectory_drivers",
        type=str,
        default="",
        help="style=driver_id CSV folders to plot (e.g. conservative=T9,...).",
    )
    ap.add_argument(
        "--legend",
        type=str,
        default="",
        help="style=legend text only (e.g. conservative=Conservative (T16),...).",
    )
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--t_min", type=float, default=20.0)
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit("[ERR] not a directory: {}".format(data_dir))

    default_drv = "conservative=T16,neutral=T15,aggressive=T19"
    if args.drivers.strip():
        traj_map = _parse_style_map(args.drivers, "--drivers", upper_values=True)
        leg_map = {s: "{} ({})".format(s.capitalize(), traj_map[s]) for s in traj_map}
    else:
        traj_s = args.trajectory_drivers.strip() or default_drv
        traj_map = _parse_style_map(traj_s, "--trajectory_drivers", upper_values=True)
        if args.legend.strip():
            leg_map = _parse_style_map(args.legend, "--legend", upper_values=False)
        else:
            leg_map = {s: "{} ({})".format(s.capitalize(), traj_map[s]) for s in traj_map}

    styles = _ordered_styles(traj_map)
    if not styles:
        raise SystemExit("[ERR] no trajectory drivers configured")
    for s in styles:
        if s not in leg_map:
            leg_map[s] = "{} ({})".format(s.capitalize(), traj_map[s])

    series: Dict[str, Dict] = {}
    for style in styles:
        driver = traj_map[style]
        csv_path = data_dir / driver / "driving_data.csv"
        if not csv_path.is_file():
            raise SystemExit("[ERR] missing {}".format(csv_path))
        s = _load_post20_series(csv_path, args.t_min)
        if not s["t"]:
            raise SystemExit("[ERR] no rows with sim_time_s >= {:.1f} in {}".format(args.t_min, csv_path))
        series[style] = {
            "legend": leg_map[style],
            **s,
            "trajectory_driver": driver,
            "color": _STYLE_COLORS[style],
        }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for style in styles:
        s = series[style]
        lbl_gap = "{} | mean gap={:.1f}m".format(s["legend"], s["mean_gap"])
        axes[0].plot(s["t"], s["gap"], lw=2.0, color=s["color"], label=lbl_gap)
        axes[1].plot(s["t"], s["v"], lw=1.8, color=s["color"], label=s["legend"])
        axes[2].plot(s["t"], s["a"], lw=1.4, color=s["color"], label=s["legend"])

    for ax in axes:
        ax.axvline(args.t_min, color="k", ls="--", lw=1.0, alpha=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlim(left=args.t_min)

    axes[0].set_ylabel("Distance headway (m)")
    axes[1].set_ylabel("Ego speed (m/s)")
    axes[2].set_ylabel("Ego acceleration (m/s^2)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[0].set_title(
        "Typical Following Style Comparison (Post-takeover: sim_time_s >= {:.0f}s)".format(args.t_min)
    )
    for ax in axes:
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print("saved:", out)
    for style in styles:
        print(
            "  {}: trajectory={}  legend={!r}".format(
                style, series[style]["trajectory_driver"], series[style]["legend"]
            )
        )


if __name__ == "__main__":
    main()
