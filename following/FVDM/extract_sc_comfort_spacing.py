# -*- coding: utf-8 -*-
"""
Extract per-subject comfort following spacing ``s_c`` (steady-state median headway).

This entry point forwards to ``calibrate_fvdm_from_following.extract_sc_comfort_spacing_cli`` so
logic lives in one module with FVDM calibration. See module docstrings there.

Example::

  python3 following/FVDM/extract_sc_comfort_spacing.py \
    --data_dir following/outputs/following_calibrated \
    --out_csv following/outputs/s_c_per_driver.csv
"""
from __future__ import print_function

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from calibrate_fvdm_from_following import extract_sc_comfort_spacing_cli  # noqa: E402


if __name__ == "__main__":
    sys.exit(extract_sc_comfort_spacing_cli() or 0)
