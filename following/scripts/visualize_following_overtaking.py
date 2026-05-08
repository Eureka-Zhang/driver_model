#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跟驰 / 超车 driving_data.csv 统一可视化。

方案概要
--------
1. 数据：与仓库内其他脚本一致，读取 ``driving_data.csv``；时间轴优先 ``sim_time_s``，否则 ``timestamp``。
2. 哨兵：``time_headway`` / ``ttc`` 为 999 或无效时在图中以 NaN 断开，避免纵轴被撑爆。
3. 模式：
   - **跟驰**：纵向位置、速度、加速度、车间距；THW 与 TTC 同轴对比；``ego_pos_y``（及 ``lead_pos_y``）观察车道保持。
   - **超车**：平面轨迹 (ego / lead)、纵向 x、横向 y、速度、加速度；若有 ``steer`` 则叠在横向 y 子图次轴（不画 headway / THW / TTC）。
4. 自动判别：路径名含 ``_o`` / ``overtaking`` → 超车布局；含 ``_f`` / ``following`` → 跟驰；否则根据 ``ego_pos_y`` 标准差启发式（默认阈值 0.35 m）。

单文件（弹窗）::

    python3 following/scripts/visualize_following_overtaking.py /path/to/driving_data.csv

批量（无界面，递归写 PNG）::

    python3 following/scripts/visualize_following_overtaking.py /path/to/session_root --batch \\
        -o following/outputs/pictures/following_overtaking_viz

强制模式::

    python3 following/scripts/visualize_following_overtaking.py data.csv --mode overtaking
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from visualize_data import maybe_smooth  # noqa: E402


def _configure_matplotlib_style() -> None:
    """Prefer a CJK-capable sans font so Chinese titles render; avoid broken minus sign."""
    # Matplotlib picks the first family that exists on the system (see font.sans-serif fallback).
    matplotlib.rcParams['font.sans-serif'] = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Serif CJK SC',
        'Source Han Sans SC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Microsoft YaHei',
        'SimHei',
        'Arial Unicode MS',
        'DejaVu Sans',
    ]
    matplotlib.rcParams['axes.unicode_minus'] = False


def _discover_driving_data_csvs(root: Path) -> List[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob('driving_data.csv') if p.is_file())


def default_out_dir() -> Path:
    return _SCRIPT_DIR.parent / 'outputs' / 'pictures' / 'following_overtaking_viz'


def infer_mode_from_path(csv_path: Path) -> Optional[str]:
    s = str(csv_path.resolve()).lower()
    if 'overtaking' in s or re.search(r'(^|[^a-z0-9])_o([^a-z0-9]|$)', s) or re.search(r'exp[123]_o', s):
        return 'overtaking'
    if 'following' in s or re.search(r'(^|[^a-z0-9])_f([^a-z0-9]|$)', s):
        return 'following'
    return None


def infer_mode_from_dataframe(df: pd.DataFrame, *, lateral_std_threshold_m: float) -> str:
    if 'ego_pos_y' not in df.columns:
        return 'following'
    y = pd.to_numeric(df['ego_pos_y'], errors='coerce').to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 'following'
    if float(np.std(y)) >= lateral_std_threshold_m:
        return 'overtaking'
    return 'following'


def _mask_sentinel(series: np.ndarray, *, sentinel: float = 999.0, positive_only: bool = False) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    out = x.copy()
    bad = ~np.isfinite(out)
    bad |= np.isclose(out, sentinel, rtol=0, atol=1e-3)
    if positive_only:
        bad |= out <= 0.0
    out[bad] = np.nan
    return out


def _resolve_png_out(
    csv_path: Path,
    out_root: Path,
    *,
    mirror_under: Optional[Path],
    flat: bool,
) -> Path:
    csv_path = csv_path.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if flat:
        return out_root / f'{csv_path.parent.name}_{csv_path.stem}_fo.png'
    if mirror_under is None:
        return out_root / f'{csv_path.parent.name}_{csv_path.stem}_fo.png'
    rel = csv_path.relative_to(mirror_under.resolve())
    target = out_root / rel.with_name(rel.stem + '_fo.png')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_following(
    df: pd.DataFrame,
    *,
    smooth: bool,
    title_suffix: str = '',
):
    import matplotlib.pyplot as plt

    time_col = 'sim_time_s' if 'sim_time_s' in df.columns else 'timestamp'
    t = df[time_col].to_numpy(dtype=float)

    fig, axs = plt.subplots(6, 1, figsize=(10, 18), dpi=120)
    sm = smooth

    ego_x = maybe_smooth(df['ego_pos_x'], sm, frac_window=0.04)
    lead_x = maybe_smooth(df['lead_pos_x'], sm, frac_window=0.04)
    axs[0].plot(t, ego_x, label='Ego X', color='#1f77b4', linewidth=1.2)
    axs[0].plot(t, lead_x, label='Lead X', color='#ff7f0e', linewidth=1.2)
    axs[0].set_ylabel('x (m)')
    axs[0].set_title('跟驰 — 纵向位置' + title_suffix)
    axs[0].legend(loc='upper right', fontsize=8)
    axs[0].grid(True, linestyle='--', alpha=0.6)

    ego_v = maybe_smooth(df['ego_speed'], sm, frac_window=0.04)
    lead_v = maybe_smooth(df['lead_speed'], sm, frac_window=0.04)
    axs[1].plot(t, ego_v, label='Ego speed', color='#1f77b4', linewidth=1.2)
    axs[1].plot(t, lead_v, label='Lead speed', color='#ff7f0e', linewidth=1.2)
    axs[1].set_ylabel('m/s')
    axs[1].legend(loc='upper right', fontsize=8)
    axs[1].grid(True, linestyle='--', alpha=0.6)

    ego_a = maybe_smooth(df['ego_acceleration'], sm, frac_window=0.06)
    axs[2].plot(t, ego_a, label='Ego accel', color='#d62728', linewidth=1.0)
    axs[2].set_ylabel('m/s²')
    axs[2].legend(loc='upper right', fontsize=8)
    axs[2].grid(True, linestyle='--', alpha=0.6)

    dh = maybe_smooth(df['distance_headway'], sm, frac_window=0.04)
    axs[3].plot(t, dh, label='Headway', color='#2ca02c', linewidth=1.2)
    axs[3].set_ylabel('m')
    axs[3].legend(loc='upper right', fontsize=8)
    axs[3].grid(True, linestyle='--', alpha=0.6)

    if 'time_headway' in df.columns:
        thw = _mask_sentinel(
            np.asarray(maybe_smooth(df['time_headway'], sm, frac_window=0.04), dtype=float),
            positive_only=True,
        )
        axs[4].plot(t, thw, label='THW', color='#9467bd', linewidth=1.0)
    if 'ttc' in df.columns:
        ttc = _mask_sentinel(
            np.asarray(maybe_smooth(df['ttc'], sm, frac_window=0.04), dtype=float),
            positive_only=True,
        )
        axs[4].plot(t, ttc, label='TTC', color='#8c564b', linewidth=1.0)
    axs[4].set_ylabel('s')
    axs[4].set_title('THW / TTC（哨兵已屏蔽）')
    axs[4].legend(loc='upper right', fontsize=8)
    axs[4].grid(True, linestyle='--', alpha=0.6)

    if 'ego_pos_y' in df.columns:
        ey = maybe_smooth(df['ego_pos_y'], sm, frac_window=0.04)
        axs[5].plot(t, ey, label='Ego Y', color='#17becf', linewidth=1.0)
    if 'lead_pos_y' in df.columns:
        ly = maybe_smooth(df['lead_pos_y'], sm, frac_window=0.04)
        axs[5].plot(t, ly, label='Lead Y', color='#bcbd22', linewidth=1.0, alpha=0.8)
    axs[5].set_xlabel('Time (s)')
    axs[5].set_ylabel('y (m)')
    axs[5].legend(loc='upper right', fontsize=8)
    axs[5].grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()
    return fig


def plot_overtaking(
    df: pd.DataFrame,
    *,
    smooth: bool,
    title_suffix: str = '',
):
    import matplotlib.pyplot as plt

    time_col = 'sim_time_s' if 'sim_time_s' in df.columns else 'timestamp'
    t = df[time_col].to_numpy(dtype=float)
    sm = smooth

    # constrained_layout handles twinx() better than tight_layout (matplotlib warns otherwise).
    use_constrained = True
    try:
        fig = plt.figure(figsize=(11, 14), dpi=120, layout='constrained')
    except TypeError:
        fig = plt.figure(figsize=(11, 14), dpi=120)
        use_constrained = False
    gs = fig.add_gridspec(5, 1, height_ratios=[1.15, 1, 1, 1, 1])
    ax_traj = fig.add_subplot(gs[0, 0])

    ego_x = maybe_smooth(df['ego_pos_x'], sm, frac_window=0.04).astype(float)
    ego_y = maybe_smooth(df['ego_pos_y'], sm, frac_window=0.04).astype(float)
    lead_x = maybe_smooth(df['lead_pos_x'], sm, frac_window=0.04).astype(float)
    lead_y = maybe_smooth(df['lead_pos_y'], sm, frac_window=0.04).astype(float)
    ax_traj.plot(ego_x, ego_y, label='Ego', color='#1f77b4', linewidth=1.4)
    ax_traj.plot(lead_x, lead_y, label='Lead', color='#ff7f0e', linewidth=1.2, alpha=0.9)
    ax_traj.set_xlabel('x (m)')
    ax_traj.set_ylabel('y (m)')
    ax_traj.set_title('超车 — 平面轨迹（纵横比自动）' + title_suffix)
    ax_traj.legend(loc='best', fontsize=8)
    ax_traj.grid(True, linestyle='--', alpha=0.6)
    ax_traj.set_aspect('auto')

    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.plot(t, ego_x, label='Ego X', color='#1f77b4', linewidth=1.0)
    ax_x.plot(t, lead_x, label='Lead X', color='#ff7f0e', linewidth=1.0)
    ax_x.set_ylabel('x (m)')
    ax_x.legend(loc='upper right', fontsize=8)
    ax_x.grid(True, linestyle='--', alpha=0.6)

    ax_y = fig.add_subplot(gs[2, 0])
    ax_y.plot(t, ego_y, label='Ego Y', color='#1f77b4', linewidth=1.0)
    ax_y.plot(t, lead_y, label='Lead Y', color='#ff7f0e', linewidth=1.0)
    ax_y.set_ylabel('y (m)')
    ax_y.legend(loc='upper right', fontsize=8)
    ax_y.grid(True, linestyle='--', alpha=0.6)

    ax_v = fig.add_subplot(gs[3, 0])
    ax_v.plot(t, maybe_smooth(df['ego_speed'], sm, frac_window=0.04), label='Ego speed', color='#1f77b4', linewidth=1.0)
    ax_v.plot(t, maybe_smooth(df['lead_speed'], sm, frac_window=0.04), label='Lead speed', color='#ff7f0e', linewidth=1.0)
    ax_v.set_ylabel('m/s')
    ax_v.legend(loc='upper right', fontsize=8)
    ax_v.grid(True, linestyle='--', alpha=0.6)

    ax_a = fig.add_subplot(gs[4, 0])
    ax_a.plot(t, maybe_smooth(df['ego_acceleration'], sm, frac_window=0.06), label='Ego accel', color='#d62728', linewidth=1.0)
    ax_a.set_xlabel('Time (s)')
    ax_a.set_ylabel('m/s²')
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.grid(True, linestyle='--', alpha=0.6)

    if 'steer' in df.columns:
        ax_st = ax_y.twinx()
        st = maybe_smooth(df['steer'], sm, frac_window=0.08)
        ax_st.plot(t, st, color='gray', alpha=0.45, linewidth=0.8, label='Steer')
        ax_st.set_ylabel('Steer', color='gray', fontsize=8)
        ax_st.tick_params(axis='y', labelcolor='gray')

    if not use_constrained:
        fig.subplots_adjust(hspace=0.35, wspace=0.28)
    return fig


def run_one(
    csv_path: Path,
    *,
    mode: str,
    smooth: bool,
    save: bool,
    show: bool,
    out_path: Optional[Path],
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    _configure_matplotlib_style()

    df = pd.read_csv(csv_path)
    path_guess = infer_mode_from_path(csv_path)
    if mode == 'auto':
        m = path_guess or infer_mode_from_dataframe(df, lateral_std_threshold_m=0.35)
    else:
        m = mode

    title_suffix = f'  [{csv_path.parent.name}]'
    if m == 'overtaking':
        fig = plot_overtaking(df, smooth=smooth, title_suffix=title_suffix)
    else:
        fig = plot_following(df, smooth=smooth, title_suffix=title_suffix)

    written: Optional[Path] = None
    if save and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
        written = out_path
        print(f'Saved {out_path}  (mode={m})')

    if show:
        plt.show()

    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description='Visualize car-following vs overtaking driving_data.csv')
    parser.add_argument('path', type=str, help='driving_data.csv or a directory (recursive batch).')
    parser.add_argument(
        '--mode',
        choices=('auto', 'following', 'overtaking'),
        default='auto',
        help='Plot layout (default: infer from path / lateral motion).',
    )
    parser.set_defaults(smooth=True)
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--smooth', dest='smooth', action='store_true')
    g.add_argument('--no-smooth', dest='smooth', action='store_false')
    parser.add_argument('--batch', action='store_true', help='Single file: save only, no window. Implied for directories.')
    parser.add_argument(
        '-o',
        '--out-dir',
        type=str,
        default=None,
        help=f'Output directory for PNGs (default: {default_out_dir()})',
    )
    parser.add_argument(
        '--flat-names',
        action='store_true',
        help='Flat filenames under out-dir instead of mirroring subtree.',
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else default_out_dir()

    is_batch = args.batch or path.is_dir()
    if is_batch:
        matplotlib.use('Agg')

    if path.is_file():
        if args.batch:
            print('Note: --batch saves without displaying; output under -o / default.')
        out_png = out_root / f'{path.parent.name}_{path.stem}_fo.png'
        run_one(path, mode=args.mode, smooth=args.smooth, save=True, show=not is_batch, out_path=out_png)
        return 0

    if not path.is_dir():
        print(f'Not a file or directory: {path}', file=sys.stderr)
        return 1

    csvs = _discover_driving_data_csvs(path)
    if not csvs:
        print(f'No driving_data.csv under {path}', file=sys.stderr)
        return 1

    matplotlib.use('Agg')
    ok = 0
    mirror = None if args.flat_names else path
    for fp in csvs:
        try:
            png = _resolve_png_out(fp, out_root, mirror_under=mirror, flat=args.flat_names)
            run_one(fp, mode=args.mode, smooth=args.smooth, save=True, show=False, out_path=png)
            ok += 1
        except Exception as e:
            print(f'SKIP {fp}: {e}', file=sys.stderr)
    print(f'Done: {ok}/{len(csvs)} written under {out_root}')
    return 0 if ok == len(csvs) else 2


if __name__ == '__main__':
    raise SystemExit(main())
