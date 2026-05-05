#!/usr/bin/env python3
"""
Recursively find driving_data CSV files under a root directory and generate the same
four-panel plots as visualize_data.py.

Uses a non-interactive matplotlib backend so no windows open during batch runs.
PNG files are written under ``<out-dir>/<ROOT_BASENAME>/...`` where ROOT_BASENAME is
the last path component of ROOT (e.g. ``outputs`` when ROOT is ``.../following/outputs``),
then mirroring paths inside ROOT unless ``--flat-names`` is set.

Run with Python 3, for example::

    python3 following/scripts/batch_visualize_driving_data.py [ROOT]
    
    python3 /home/zwx/driver_model/following/scripts/batch_visualize_driving_data.py /home/zwx/driver_model/following/outputs/personalized_no_driver_common_lead

"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use('Agg')

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from visualize_data import visualize_driving_csv  # noqa: E402


DRIVING_DATA_NAME = 'driving_data.csv'


def find_driving_data_csv_files(root: Path) -> List[Path]:
    root = root.resolve()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob(DRIVING_DATA_NAME) if p.is_file())


def main() -> int:
    following_dir = Path(__file__).resolve().parent.parent
    default_root = following_dir / 'outputs'

    parser = argparse.ArgumentParser(
        description=f'Batch plot every `{DRIVING_DATA_NAME}` found under ROOT (recursive).'
    )
    parser.add_argument(
        'root',
        nargs='?',
        default=str(default_root),
        type=str,
        help=f'Directory to search recursively (default: {default_root})',
    )
    parser.add_argument(
        '-o',
        '--out-dir',
        type=str,
        default=None,
        help='PNG parent directory (default: following/outputs/pictures); PNGs go under '
        '<OUT_DIR>/<ROOT_LAST_NAME>/',
    )
    parser.set_defaults(smooth=True)
    sm_group = parser.add_mutually_exclusive_group()
    sm_group.add_argument('--smooth', dest='smooth', action='store_true', help='Apply smoothing (default).')
    sm_group.add_argument('--no-smooth', dest='smooth', action='store_false', help='Raw series, no filter.')
    parser.add_argument(
        '--flat-names',
        action='store_true',
        help='Write PNGs as {parent}_{stem}.png flat in OUT_DIR (may overwrite unrelated runs).',
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    pictures_base = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else following_dir / 'outputs' / 'pictures'
    )
    root_tag = root.name or 'batch'
    pictures_dir = pictures_base / root_tag

    csv_files = find_driving_data_csv_files(root)
    if not csv_files:
        print(f'No `{DRIVING_DATA_NAME}` under {root}')
        return 1

    print(f'Found {len(csv_files)} file(s); writing under {pictures_dir}')
    ok = 0
    for csv_path in csv_files:
        try:
            visualize_driving_csv(
                csv_path,
                smooth=args.smooth,
                save=True,
                show=False,
                pictures_dir=pictures_dir,
                flat_output=args.flat_names,
                path_relative_root=None if args.flat_names else root,
            )
            ok += 1
        except Exception as e:
            print(f'SKIP {csv_path}: {e}')

    print(f'Done: {ok}/{len(csv_files)} saved.')
    return 0 if ok == len(csv_files) else 2


if __name__ == '__main__':
    raise SystemExit(main())
