#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot typical style comparison for T2/T10/T12 after takeover (sim_time_s >= 20s).

Usage:
    python3 following/scripts/plot_typical_style_post20.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    base = Path("/home/zwx/driver_model/following/outputs/residual_gru_takeover_20s")
    profiles = {
        "T2": "Conservative (T2)",
        "T10": "Neutral (T10)",
        "T11": "Aggressive (T11)",
    }
    colors = {"T2": "#1f77b4", "T10": "#2ca02c", "T11": "#d62728"}

    series = {}
    for driver, label in profiles.items():
        csv_path = base / driver / "driving_data.csv"
        t, gap, v, a = [], [], [], []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tt = float(row["sim_time_s"])
                    if tt < 20.0:
                        continue
                    gg = float(row["distance_headway"])
                    vv = float(row.get("ego_v_long") or row.get("ego_speed"))
                    aa = float(row.get("ego_a_long") or row.get("ego_acceleration"))
                except Exception:
                    continue
                t.append(tt)
                gap.append(gg)
                v.append(vv)
                a.append(aa)

        mean_gap = (sum(gap) / len(gap)) if gap else float("nan")
        series[driver] = {
            "label": label,
            "t": t,
            "gap": gap,
            "v": v,
            "a": a,
            "mean_gap": mean_gap,
        }

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for driver in profiles:
        s = series[driver]
        lbl = "{} | mean gap={:.1f}m".format(s["label"], s["mean_gap"])
        axes[0].plot(s["t"], s["gap"], lw=2.0, color=colors[driver], label=lbl)
        axes[1].plot(s["t"], s["v"], lw=1.8, color=colors[driver], label=s["label"])
        axes[2].plot(s["t"], s["a"], lw=1.4, color=colors[driver], label=s["label"])

    for ax in axes:
        ax.axvline(20.0, color="k", ls="--", lw=1.0, alpha=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlim(left=20.0)

    axes[0].set_ylabel("Distance headway (m)")
    axes[1].set_ylabel("Ego speed (m/s)")
    axes[2].set_ylabel("Ego acceleration (m/s^2)")
    axes[2].set_xlabel("Simulation time (s)")
    axes[0].set_title("Typical Following Style Comparison (Post-takeover: sim_time_s >= 20s)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = Path(
        "/home/zwx/driver_model/following/outputs/pictures/residual_gru_takeover_20s/typical_style_comparison_T2_T10_T11_post20s.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print("saved:", out)


if __name__ == "__main__":
    main()
