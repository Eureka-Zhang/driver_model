# -*- coding: utf-8 -*-
"""Draw Gain-Residual GRU neural network architecture (v3)."""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(__file__), "gru_architecture.png")

# Default hyperparameters from train_bc_gru_residual_v3.py
SEQ_LEN = 20
D_IN = 7
HIDDEN = 128
N_LAYERS = 2
DROPOUT = 0.1
FEATURES = [
    "v", "a", "gap", "rel_v", "v_lead", "inv_ttc", "inv_thw",
]


def _box(ax, xy, wh, text, fc="#f7f9fc", ec="#334155", fs=9, lw=1.3):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return p


def _layer_stack(ax, x, y, w, h, n, color="#dbeafe", label=""):
    """Draw stacked rectangles to suggest depth (GRU layers)."""
    dy = h * 0.06
    for i in range(n):
        off = i * dy
        r = Rectangle(
            (x + off, y + off), w, h,
            linewidth=1.0, edgecolor="#2563eb", facecolor=color, alpha=0.85 - i * 0.15,
        )
        ax.add_patch(r)
    cx = x + w / 2 + (n - 1) * dy / 2
    cy = y + h / 2 + (n - 1) * dy / 2
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9, fontweight="bold")


def _arrow(ax, p0, p1, text="", color="#64748b", rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
        color=color, connectionstyle="arc3,rad={}".format(rad),
    ))
    if text:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2
        ax.text(mx, my + 0.012, text, ha="center", va="bottom", fontsize=fs, color=color)


def main():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.97,
        "Gain-Residual GRU Network (gain_residual_gru_v3)",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )
    ax.text(
        0.5, 0.925,
        "train_bc_gru_residual_v3.py  ·  ResidualGRUPolicy inference uses the same topology",
        ha="center", va="top", fontsize=9, color="#475569",
    )

    # ---- Input ----
    feat_txt = "\n".join(["{}: {}".format(i + 1, nm) for i, nm in enumerate(FEATURES)])
    _box(
        ax, (0.03, 0.48), (0.16, 0.32),
        "Input window\n(batch, {}, {})\n\n".format(SEQ_LEN, D_IN) + feat_txt,
        fc="#eef2ff", fs=7.5,
    )

    _box(
        ax, (0.03, 0.30), (0.16, 0.12),
        "Z-score\n(x - mean) / std\n(per training stats)",
        fc="#f1f5f9", fs=8,
    )

    _arrow(ax, (0.11, 0.48), (0.11, 0.42))

    # ---- GRU backbone ----
    _layer_stack(
        ax, 0.24, 0.42, 0.18, 0.36, N_LAYERS,
        color="#bfdbfe",
        label="GRU\nhidden={}\nlayers={}\ndropout={}".format(HIDDEN, N_LAYERS, DROPOUT),
    )
    ax.text(0.33, 0.38, "batch_first=True\nunroll over time",
            ha="center", fontsize=7.5, color="#1e40af")

    _box(
        ax, (0.46, 0.52), (0.14, 0.16),
        "Last timestep\nhidden state\nh_T  (128-d)",
        fc="#fff7ed", ec="#c2410c", fs=8.5,
    )

    _arrow(ax, (0.19, 0.58), (0.24, 0.58), "(B,20,7)")
    _arrow(ax, (0.42, 0.60), (0.46, 0.60), "out[:,-1,:]")

    # ---- MLP head ----
    _box(ax, (0.64, 0.58), (0.12, 0.10), "Linear\n128 → 64", fc="#fef3c7", ec="#b45309", fs=8)
    _box(ax, (0.64, 0.44), (0.12, 0.08), "ReLU", fc="#fef9c3", ec="#ca8a04", fs=8)
    _box(ax, (0.64, 0.30), (0.12, 0.10), "Linear\n64 → 2", fc="#fef3c7", ec="#b45309", fs=8)

    _arrow(ax, (0.60, 0.60), (0.64, 0.63))
    _arrow(ax, (0.70, 0.58), (0.70, 0.52))
    _arrow(ax, (0.70, 0.44), (0.70, 0.40))

    # ---- Output squashing ----
    _box(
        ax, (0.82, 0.52), (0.14, 0.14),
        r"$\alpha_{raw}$" " → sigmoid map\n"
        r"$\alpha \in [-0.5,\,2.0]$" "\n(gain on IDM)",
        fc="#fee2e2", ec="#dc2626", fs=8,
    )
    _box(
        ax, (0.82, 0.30), (0.14, 0.14),
        r"$\Delta a_{raw}$" " → tanh × 2\n"
        r"$\Delta a \in [-2,\,2]$" "\n(residual accel.)",
        fc="#fee2e2", ec="#dc2626", fs=8,
    )

    _arrow(ax, (0.76, 0.35), (0.82, 0.59), "ch.0", rad=0.15)
    _arrow(ax, (0.76, 0.35), (0.82, 0.37), "ch.1", rad=-0.15)

    # ---- IDM fusion (external) ----
    _box(
        ax, (0.46, 0.12), (0.16, 0.12),
        "Frozen IDM\na_IDM(v, v_lead, gap)\nfrom idm.json",
        fc="#dcfce7", ec="#15803d", fs=8,
    )
    _box(
        ax, (0.66, 0.10), (0.26, 0.14),
        r"$a_{pred}=\mathrm{clip}\!\left[(1+\alpha)\,a_{IDM}+\Delta a\right]$",
        fc="#fce7f3", ec="#be185d", fs=10,
    )

    _arrow(ax, (0.89, 0.52), (0.72, 0.24), r"$\alpha$", rad=0.2, color="#dc2626")
    _arrow(ax, (0.89, 0.30), (0.78, 0.18), r"$\Delta a$", rad=-0.1, color="#dc2626")
    _arrow(ax, (0.54, 0.12), (0.66, 0.14), r"$a_{IDM}$", color="#15803d")

    # ---- Training side note ----
    _box(
        ax, (0.03, 0.06), (0.38, 0.16),
        "Training (closed-loop sub-segment, len=25)\n"
        "· GRU hidden h carried across steps\n"
        "· State anchoring: blend v_cf, gap_cf toward real log\n"
        "· Loss = MSE(a) + λ_v MSE(v) + λ_g MSE(gap)\n"
        "         + λ_α α² + λ_δ Δa²\n"
        "· Optimizer: Adam (lr=5e-4, weight_decay=1e-5)",
        fc="#f8fafc", ec="#94a3b8", fs=7.5,
    )

    # Parameter count annotation
    # GRU params approx: 3*(d_in*h + h*h + h) per layer for input layer, 3*(h*h+h) for others
    # Rough: layer1 ~ 3*(7*128+128*128+128)=52k, layer2 ~ 3*(128*128+128)=49k, head 128*64+64+64*2+2 ~ 8354
    ax.text(
        0.88, 0.06,
        "~105k trainable params\n(default config)",
        ha="center", va="bottom", fontsize=8, color="#475569",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f1f5f9", ec="#cbd5e1"),
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]", OUT)


if __name__ == "__main__":
    main()
