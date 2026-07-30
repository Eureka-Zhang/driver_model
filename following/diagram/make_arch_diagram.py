# -*- coding: utf-8 -*-
"""
生成模型架构图并保存为 PNG。
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out_dir = Path(__file__).resolve().parents[1] / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "arch_diagram.png"

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

def box(x, y, w, h, text, fc='#eeeeff'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", linewidth=1, facecolor=fc, edgecolor='k')
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', wrap=True)
    return rect

def arrow(x0,y0,x1,y1):
    arr = FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='->',mutation_scale=12,linewidth=1)
    ax.add_patch(arr)
    return arr

# Layout boxes
b_input = box(0.2, 3.2, 2.4, 1.2, "输入窗口\n(seq_len x 7 特征)\n(v, a, gap, rel_v, lead_v, inv_ttc, inv_thw)")
box_norm = box(0.2, 1.8, 2.4, 1.2, "归一化\n(mean/std)", fc='#fff2cc')

b_model = box(3.0, 2.5, 2.6, 2.2, "GRU (layers, hidden)\nFC -> ReLU -> FC\n输出: alpha_raw, delta_raw", fc='#e6ffe6')

b_idm = box(6.0, 3.2, 2.6, 1.2, "IDM 模块\n参数(v0,s0,a,b,T,delta)\n计算 a_IDM", fc='#ffe6e6')

b_combine = box(6.0, 1.6, 2.6, 1.2, "合成: a_pred = (1+alpha)*a_IDM + delta\nclip(a_pred)", fc='#e8e8ff')

b_vehicle = box(9.0, 1.8, 2.6, 2.2, "闭环动力学\nv_free, gap_free\n锚定: v_next=(1-beta)*v_free+beta*v_real\n更新历史窗口", fc='#fff0e6')

b_losses = box(3.0, 0.2, 5.6, 1.0, "损失:\nL_res=(a_pred-a_real)^2\n+ lambda_v L_v + lambda_gap L_g\n+ lambda_alpha alpha^2 + lambda_delta delta^2", fc='#f2f2f2')

# Arrows
arrow(2.6, 3.8, 3.0, 4.0)  # input -> model
arrow(1.4, 2.95, 1.4, 2.0) # input -> norm
arrow(2.6, 2.4, 3.0, 3.0)
arrow(5.6, 3.8, 6.0, 3.8)  # model -> idm (alpha/delta flow)
arrow(5.6, 3.0, 6.0, 2.4)  # idm -> combine
arrow(8.6, 2.6, 9.0, 2.6)  # combine -> vehicle
arrow(5.2, 1.6, 3.6, 1.6)  # combine -> losses
arrow(5.9, 1.1, 4.2, 1.1)  # vehicle -> losses (v_free/gap_free)

# Closed-loop arrow back to input (feature update)
arrow(11.6, 2.8, 11.6, 4.2)
arrow(11.6, 4.2, 3.8, 4.2)
arrow(3.8, 4.2, 2.6, 4.2)

# Title
ax.text(6, 5.6, "Gain + Residual GRU 架构（训练闭环 + 状态锚定）", ha='center', va='center', fontsize=12, fontweight='bold')

plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
print("Saved:", out_path)

if __name__ == '__main__':
    pass
