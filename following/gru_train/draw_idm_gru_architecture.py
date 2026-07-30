# -*- coding: utf-8 -*-
"""Draw IDM calibration + gain-residual GRU architecture (v3)."""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "idm_gru_architecture.png")


def _box(ax, xy, wh, text, fc="#f7f9fc", ec="#334155", fs=9, lw=1.2, pad=0.012):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return p


def _arrow(ax, p0, p1, text="", color="#64748b", rad=0.0):
    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        connectionstyle="arc3,rad={}".format(rad),
    )
    ax.add_patch(arr)
    if text:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.015, text, ha="center", va="bottom", fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "IDM-GRU Car-Following Architecture (gain + residual v3)",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    # ---- Stage 1: IDM calibration ----
    ax.text(0.25, 0.90, "Stage 1 · Per-driver IDM calibration", ha="center", fontsize=11, fontweight="bold")
    _box(
        ax,
        (0.04, 0.72),
        (0.18, 0.12),
        "Following CSV\n(v, v_lead, gap, a_real)",
        fc="#eef2ff",
    )
    _box(
        ax,
        (0.28, 0.68),
        (0.22, 0.20),
        "Treiber IDM\n"
        r"$a_{IDM}=a\!\left[1-\left(\frac{v}{v_0}\right)^\delta-\left(\frac{s^*}{s}\right)^2\right]$"
        "\n"
        r"$s^*=s_0+vT+\frac{v(v-v_{lead})}{2\sqrt{ab}}$",
        fc="#ffffff",
        fs=8,
    )
    _box(
        ax,
        (0.28, 0.58),
        (0.22, 0.08),
        "Learnable params: v0, s0, a, b, T, (delta)\nOptimizer: Adam (multi-restart)",
        fc="#f1f5f9",
        fs=8,
    )
    _box(
        ax,
        (0.54, 0.72),
        (0.16, 0.12),
        "idm.json\n(per driver T*)",
        fc="#dcfce7",
        ec="#15803d",
    )
    _arrow(ax, (0.22, 0.78), (0.28, 0.78))
    _arrow(ax, (0.50, 0.78), (0.54, 0.78), "MSE vs a_real")

    # ---- Stage 2: GRU policy ----
    ax.text(0.72, 0.90, "Stage 2 · Gain-residual GRU policy", ha="center", fontsize=11, fontweight="bold")

    _box(
        ax,
        (0.62, 0.72),
        (0.20, 0.12),
        "State window (seq_len=20)\n"
        "7-dim features / step:\n"
        "v, a, gap, rel_v, v_lead,\n"
        "inv_ttc, inv_thw",
        fc="#eef2ff",
        fs=8,
    )
    _box(
        ax,
        (0.62, 0.52),
        (0.20, 0.14),
        "Z-score normalize\n(per training set stats)",
        fc="#f1f5f9",
        fs=8,
    )
    _box(
        ax,
        (0.62, 0.30),
        (0.20, 0.16),
        "GRU × 2 layers\nhidden=128, dropout=0.1\n(last hidden state)",
        fc="#fff7ed",
        ec="#c2410c",
    )
    _box(
        ax,
        (0.84, 0.30),
        (0.12, 0.16),
        "MLP head\nLinear(128→64)\nReLU\nLinear(64→2)",
        fc="#fff7ed",
        ec="#c2410c",
        fs=8,
    )
    _box(
        ax,
        (0.84, 0.12),
        (0.12, 0.12),
        r"$\alpha\in[-0.5,2.0]$" "\n"
        r"$\Delta a\in[-2,2]$",
        fc="#fef3c7",
        fs=8,
    )

    _arrow(ax, (0.72, 0.72), (0.72, 0.66))
    _arrow(ax, (0.72, 0.52), (0.72, 0.46))
    _arrow(ax, (0.72, 0.30), (0.84, 0.38))
    _arrow(ax, (0.90, 0.30), (0.90, 0.24))

    # IDM branch at inference
    _box(
        ax,
        (0.36, 0.30),
        (0.20, 0.14),
        "IDM forward\na_IDM(v_cf, v_lead, gap_cf)\n(frozen idm.json)",
        fc="#dcfce7",
        ec="#15803d",
        fs=8,
    )
    _arrow(ax, (0.62, 0.78), (0.46, 0.44), "load params", rad=-0.15)

    # Fusion
    _box(
        ax,
        (0.36, 0.10),
        (0.28, 0.12),
        r"$a_{pred}=\mathrm{clip}\!\left[(1+\alpha)\,a_{IDM}+\Delta a\right]$",
        fc="#fee2e2",
        ec="#b91c1c",
        fs=10,
    )
    _arrow(ax, (0.46, 0.30), (0.46, 0.22))
    _arrow(ax, (0.84, 0.18), (0.64, 0.16), r"$\alpha,\Delta a$", rad=0.1)

    # Closed-loop + anchoring (training)
    _box(
        ax,
        (0.04, 0.10),
        (0.26, 0.16),
        "Training rollout (segment_len=25)\n"
        "State anchoring:\n"
        r"$v_{cf}\leftarrow(1-\beta)v_{cf}+\beta v_{real}$" "\n"
        r"$\mathrm{gap}_{cf}\leftarrow(1-\beta)\mathrm{gap}_{cf}+\beta\mathrm{gap}_{real}$" "\n"
        r"Loss: MSE($a$)+$\lambda_v$MSE($v$)+$\lambda_g$MSE(gap)+reg($\alpha,\Delta a$)",
        fc="#f8fafc",
        fs=7.5,
    )
    _arrow(ax, (0.36, 0.16), (0.30, 0.18), "integrate v, gap", rad=0.0)
    _arrow(ax, (0.17, 0.26), (0.62, 0.72), "next window", rad=0.25, color="#94a3b8")

    ax.text(
        0.5,
        0.02,
        "Scripts: fit_idm_per_driver.py  →  train_bc_gru_residual_v3.py  →  residual_gru_policy.py",
        ha="center",
        fontsize=9,
        color="#475569",
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]", OUT)


if __name__ == "__main__":
    main()
