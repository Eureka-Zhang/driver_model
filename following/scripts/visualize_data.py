import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
python /home/zwx/driver_model/following/scripts/visualize_data.py /home/zwx/driver_model/following/outputs/following_calibrated/T12/行车/20260421_120610_198_exp1_f/driving_data.csv --no-smooth
'''

try:
    from scipy.signal import savgol_filter as _savgol_filter
except ImportError:  # optional dependency
    _savgol_filter = None


def default_pictures_dir() -> Path:
    return Path(__file__).resolve().parent.parent / 'outputs' / 'pictures'


def smooth_1d(y, frac_window=0.04):
    """Smooth 1-D samples to reduce staircase / sawtooth noise."""
    arr = np.asarray(y, dtype=float)
    n = len(arr)
    if n < 5 or not np.any(np.isfinite(arr)):
        return arr
    max_w = n if (n % 2 == 1) else (n - 1)
    win = max(5, int(round(n * frac_window)))
    if win % 2 == 0:
        win += 1
    win = min(win, max_w)
    if win < 5:
        return arr

    if _savgol_filter is not None:
        poly = min(3, win - 1)
        poly = max(1, poly)
        try:
            return _savgol_filter(arr, win, poly)
        except ValueError:
            pass

    s = pd.Series(arr)
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def maybe_smooth(series, enabled: bool, *, frac_window: float):
    if not enabled:
        return np.asarray(series, dtype=float)
    return smooth_1d(series, frac_window=frac_window)


def _resolve_output_png(
    csv_path: Path,
    pictures_dir: Path,
    *,
    flat_output: bool,
    path_relative_root: Path | None,
) -> Path:
    csv_res = csv_path.resolve()
    if flat_output:
        pictures_dir.mkdir(parents=True, exist_ok=True)
        return pictures_dir / f'{csv_res.parent.name}_{csv_res.stem}.png'
    if path_relative_root is None:
        raise ValueError('path_relative_root is required when flat_output is False')
    root_res = path_relative_root.resolve()
    rel = csv_res.relative_to(root_res)
    out_path = pictures_dir / rel.with_suffix('.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def visualize_driving_csv(
    csv_path: str | Path,
    *,
    smooth: bool = True,
    save: bool = True,
    show: bool = True,
    pictures_dir: Path | None = None,
    flat_output: bool = True,
    path_relative_root: Path | None = None,
) -> Path | None:
    """
    Plot one driving_data CSV (position / speed / acceleration / headway) and optionally save/show.

    When flat_output is False and path_relative_root is set, the PNG path mirrors the CSV path
    relative to that root under pictures_dir (nested dirs, unique filenames).
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    time_col = 'sim_time_s' if 'sim_time_s' in df.columns else 'timestamp'
    t = df[time_col].to_numpy()

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), dpi=120)
    sm = smooth

    ego_x = maybe_smooth(df['ego_pos_x'], sm, frac_window=0.04)
    lead_x = maybe_smooth(df['lead_pos_x'], sm, frac_window=0.04)
    axs[0].plot(t, ego_x, label='Ego X', color='#1f77b4', linewidth=1.5)
    axs[0].plot(t, lead_x, label='Lead X', color='#ff7f0e', linewidth=1.5)
    axs[0].set_ylabel('Position X (m)')
    axs[0].set_title('Longitudinal position vs time')
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    ego_v = maybe_smooth(df['ego_speed'], sm, frac_window=0.04)
    lead_v = maybe_smooth(df['lead_speed'], sm, frac_window=0.04)
    axs[1].plot(t, ego_v, label='Ego speed', color='#1f77b4', linewidth=1.5)
    axs[1].plot(t, lead_v, label='Lead speed', color='#ff7f0e', linewidth=1.5)
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].set_title('Speed vs time')
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.7)

    ego_a = maybe_smooth(df['ego_acceleration'], sm, frac_window=0.06)
    axs[2].plot(t, ego_a, label='Ego acceleration', color='red', linewidth=1.5)
    axs[2].set_ylabel('Acceleration (m/s^2)')
    axs[2].set_title('Ego acceleration vs time')
    axs[2].legend(loc='upper right')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    dh = maybe_smooth(df['distance_headway'], sm, frac_window=0.04)
    axs[3].plot(t, dh, label='Distance headway', color='green', linewidth=1.5)
    axs[3].axhline(
        y=0,
        color='black',
        linestyle='--',
        linewidth=1.5,
        label='Zero gap (collision reference)',
    )
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Distance (m)')
    axs[3].set_title('Distance headway vs time')
    axs[3].legend(loc='upper right')
    axs[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    out_path: Path | None = None
    if save:
        dest = pictures_dir if pictures_dir is not None else default_pictures_dir()
        out_path = _resolve_output_png(
            csv_path,
            dest,
            flat_output=flat_output,
            path_relative_root=path_relative_root,
        )
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to {out_path}')

    if show:
        plt.show()

    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot driving_data.csv time series (position, speed, accel, headway).'
    )
    parser.add_argument(
        'csv',
        metavar='CSV',
        help='Path to the CSV file to plot (e.g. driving_data.csv).',
    )
    parser.set_defaults(smooth=True)
    sm_group = parser.add_mutually_exclusive_group()
    sm_group.add_argument(
        '--smooth',
        dest='smooth',
        action='store_true',
        help='Apply smoothing to plotted series (default).',
    )
    sm_group.add_argument(
        '--no-smooth',
        dest='smooth',
        action='store_false',
        help='Plot raw CSV columns without filtering.',
    )
    args = parser.parse_args()

    visualize_driving_csv(
        args.csv,
        smooth=args.smooth,
        save=True,
        show=True,
        flat_output=True,
    )


if __name__ == '__main__':
    main()
