# -*- coding: utf-8 -*-
"""Draw per-driver IDM calibration architecture (fit_idm_per_driver.py)."""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "idm_architecture.png")


def _box(ax, xy, wh, text, fc="#f7f9fc", ec="#334155", fs=9, lw=1.2):
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


def _arrow(ax, p0, p1, text="", color="#64748b", rad=0.0, yoff=0.012):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=1.3,
        color=color, connectionstyle="arc3,rad={}".format(rad),
    ))
    if text:
        mx = (p0[0] + p1[0]) / 2
        my = (p0[1] + p1[1]) / 2 + yoff
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.96,
        "Per-Driver IDM Calibration (Differentiable Parameter Fitting)",
        ha="center", va="top", fontsize=14, fontweight="bold",
    )
    ax.text(
        0.5, 0.915,
        "fit_idm_per_driver.py  ·  Treiber-Hennecke-Helbing IDM  ·  Adam in PyTorch",
        ha="center", va="top", fontsize=9, color="#475569",
    )

    # ---- Data pipeline ----
    _box(ax, (0.03, 0.62), (0.17, 0.18),
         "Following trajectories\nsegment_*.csv /\ndriving_data.csv\n(per driver T*)",
         fc="#eef2ff")
    _box(ax, (0.03, 0.42), (0.17, 0.14),
         "Preprocessing\n· sort by sim_time_s\n· filter gap, speed\n· t ≥ min_sim_time_s\n· drop THW≈999",
         fc="#f1f5f9", fs=8)

    _box(ax, (0.24, 0.58), (0.14, 0.22),
         "State samples\n(at each row)\n\n"
         r"$v$  ego speed" "\n"
         r"$v_{lead}$  lead speed" "\n"
         r"$s$  gap (headway)" "\n"
         r"$a_{real}$  measured accel",
         fc="#eef2ff", fs=8)

    _arrow(ax, (0.20, 0.71), (0.24, 0.71))
    _arrow(ax, (0.115, 0.62), (0.115, 0.56))
    _arrow(ax, (0.115, 0.42), (0.24, 0.58), rad=0.15)

    # ---- Learnable parameters ----
    _box(ax, (0.42, 0.72), (0.20, 0.16),
         "Learnable parameters θ\n(raw → sigmoid box)\n\n"
         r"$v_0,\; s_0,\; a,\; b,\; T,\; \delta$",
         fc="#fef3c7", ec="#b45309", fs=9)
    ax.text(0.52, 0.68, "5 params if δ fixed\n6 params if δ fitted",
            ha="center", fontsize=7.5, color="#92400e")

    # ---- IDM block ----
    _box(ax, (0.40, 0.38), (0.24, 0.26),
         "IDM forward (differentiable)\n\n"
         r"$s^* = s_0 + vT + \dfrac{v(v-v_{lead})}{2\sqrt{ab}}$" "\n\n"
         r"$a_{IDM} = a\left[1-\left(\dfrac{v}{v_0}\right)^\delta"
         r"-\left(\dfrac{s^*}{s}\right)^2\right]$",
         fc="#ffffff", ec="#0f766e", fs=8.5)

    _arrow(ax, (0.38, 0.69), (0.46, 0.64), "θ")
    _arrow(ax, (0.31, 0.69), (0.42, 0.52), r"$v,v_{lead},s$")

    # ---- Output & loss ----
    _box(ax, (0.68, 0.58), (0.14, 0.14),
         r"Predicted" "\n" r"$a_{IDM}(t)$",
         fc="#ecfdf5", ec="#059669", fs=9)

    _box(ax, (0.68, 0.38), (0.14, 0.14),
         r"Target" "\n" r"$a_{real}(t)$",
         fc="#fee2e2", ec="#dc2626", fs=9)

    _box(ax, (0.86, 0.48), (0.12, 0.14),
         "Weighted MSE\n"
         r"$\mathcal{L}=\dfrac{\sum m_t(a_{IDM}-a_{real})^2}{\sum m_t}$" "\n"
         "(rest mask m_t)",
         fc="#f8fafc", fs=8)

    _arrow(ax, (0.64, 0.65), (0.68, 0.65))
    _arrow(ax, (0.31, 0.58), (0.40, 0.50), rad=-0.1)
    _arrow(ax, (0.64, 0.45), (0.68, 0.45))
    _arrow(ax, (0.75, 0.58), (0.86, 0.55))
    _arrow(ax, (0.75, 0.45), (0.86, 0.52))

    # ---- Optimization loop ----
    _box(ax, (0.40, 0.12), (0.28, 0.16),
         "Optimization loop\n"
         "· Adam on raw θ  (lr=0.04, epochs=3000)\n"
         "· n_restarts=5, keep best RMSE\n"
         "· backprop: ∂L/∂θ through IDM equations",
         fc="#fff7ed", ec="#c2410c", fs=8)

    _box(ax, (0.74, 0.12), (0.22, 0.16),
         "Output per driver\n"
         "<out_dir>/<T*>/idm.json\n"
         "idm_all_drivers.json\n"
         "idm_per_driver_summary.csv",
         fc="#dcfce7", ec="#15803d", fs=8)

    _arrow(ax, (0.92, 0.48), (0.54, 0.28), "∂L/∂θ", rad=-0.35, color="#c2410c")
    _arrow(ax, (0.54, 0.12), (0.74, 0.20), "best θ*")

    # Note: not a deep NN
    ax.text(
        0.03, 0.06,
        "Note: IDM is a physics-based parametric model (not a deep neural network).\n"
        "PyTorch provides automatic differentiation for gradient-based calibration of θ.",
        ha="left", va="bottom", fontsize=8.5, color="#64748b",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8fafc", ec="#cbd5e1"),
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]", OUT)


if __name__ == "__main__":
    main()
