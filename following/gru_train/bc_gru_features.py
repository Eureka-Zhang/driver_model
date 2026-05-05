# -*- coding: utf-8 -*-
"""
Single source for BC-GRU row parsing / feature extraction (trained in train_bc_gru.py).

All generators must import here — do not duplicate _row_value semantics.

Default feature order (longitudinal IL, uniform sim step — no ``dt_prev``):
  ego_v_long, ego_a_long, distance_headway, relative_v_long, lead_v_long,
  inv_ttc, inv_time_headway

Legacy (variable wall-clock ``timestamp`` spacing) adds ``dt_prev`` first — see ``DEFAULT_FEATURES_LEGACY``.

``inv_ttc`` / ``inv_time_headway`` match ``calibrate_following_data`` /
``clean_following_for_imitation`` (1/x with 0 when x≈999 or invalid). No ``*_valid`` flags.
"""
import math


DEFAULT_TARGETS = ["ego_a_long"]

# Uniform simulation timeline (e.g. ``sim_time_s`` / fixed 0.05 s per row): no Δt in inputs.
DEFAULT_FEATURES = [
    "ego_v_long",
    "ego_a_long",
    "distance_headway",
    "relative_v_long",
    "lead_v_long",
    "inv_ttc",
    "inv_time_headway",
]

# Original BC-GRU layout when rows use wall-clock ``timestamp``.
DEFAULT_FEATURES_LEGACY = [
    "dt_prev",
    "ego_v_long",
    "ego_a_long",
    "distance_headway",
    "relative_v_long",
    "lead_v_long",
    "inv_ttc",
    "inv_time_headway",
]


_SENTINEL_999_TOL = 1e-3


def reciprocal_inv_feature(x):
    """
    1/x for finite ``x > 0`` not near the ``999`` sentinel; else ``0``.
    Matches ``clean_following_for_imitation._reciprocal_thw_or_zero`` / calibrated CSV.
    """
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    if abs(v - 999.0) < _SENTINEL_999_TOL:
        return 0.0
    return 1.0 / v


def _parse_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_value(row, key):
    """
    Match ``train_bc_gru.BCGRU`` training preprocessing exactly:

    - ``inv_ttc`` / ``inv_time_headway``: prefer CSV column; else derive from ``ttc`` /
      ``time_headway`` via ``reciprocal_inv_feature`` (0 when sentinel / invalid).
    - If ``key`` is present in ``row``, parse as float from that column (except special keys above).
    - ``relative_speed`` / ``relative_v_long`` aliases: derive from ``lead_speed`` − ``ego_speed``
      or lead/ego long speeds when requested as ``relative_speed``.
    """
    if key == "relative_speed":
        lv = _parse_float(row.get("lead_speed"))
        ev = _parse_float(row.get("ego_speed"))
        if lv is None or ev is None:
            return None
        return lv - ev

    if key == "inv_ttc":
        vu = _parse_float(row.get("inv_ttc"))
        if vu is not None:
            return float(vu)
        tt = _parse_float(row.get("ttc"))
        if tt is not None:
            return reciprocal_inv_feature(tt)
        return None

    if key == "inv_time_headway":
        vu = _parse_float(row.get("inv_time_headway"))
        if vu is not None:
            return float(vu)
        th = _parse_float(row.get("time_headway"))
        if th is not None:
            return reciprocal_inv_feature(th)
        return None

    # Aliases matching user-facing names
    if key == "ttc_inverse":
        return _row_value(row, "inv_ttc")
    if key == "time_headway_inverse":
        return _row_value(row, "inv_time_headway")

    if key in row:
        return _parse_float(row.get(key))
    return None


def hydrate_bc_gru_row_aliases(row):
    """
    Add canonical BC-GRU column names so ``_row_value`` succeeds when CSV uses CARLA/export
    aliases (e.g. ``ego_speed`` only). Does not change parsing rules—only inserts missing /
    unparsable canon keys before ``_row_value``.
    """
    r = dict(row)

    def _pf(key):
        if key not in r:
            return None
        return _parse_float(r.get(key))

    def _need(key):
        return _pf(key) is None

    if _need("ego_v_long"):
        for s in ("ego_speed", "ego_v"):
            if _pf(s) is not None:
                r["ego_v_long"] = str(r[s]).strip()
                break

    if _need("ego_a_long"):
        for s in ("ego_acceleration", "ego_accel", "accel_long"):
            if _pf(s) is not None:
                r["ego_a_long"] = str(r[s]).strip()
                break

    if _need("lead_v_long"):
        for s in ("lead_speed",):
            if _pf(s) is not None:
                r["lead_v_long"] = str(r[s]).strip()
                break

    if _need("relative_v_long"):
        if _pf("relative_speed") is not None:
            r["relative_v_long"] = str(r["relative_speed"]).strip()
        elif _pf("lead_v_long") is not None and _pf("ego_v_long") is not None:
            r["relative_v_long"] = "{:.6f}".format(
                float(_pf("lead_v_long")) - float(_pf("ego_v_long"))
            )

    if _need("time_headway"):
        dh = _pf("distance_headway")
        ev = _pf("ego_v_long")
        if dh is not None and ev is not None:
            if float(ev) < 0.5:
                r["time_headway"] = "999.0"
            else:
                r["time_headway"] = "{:.6f}".format(float(dh) / float(ev))

    if _need("ttc"):
        dh = _pf("distance_headway")
        ego = _pf("ego_v_long")
        lv = _pf("lead_v_long")
        tset = 999.0
        if dh is not None and ego is not None and lv is not None:
            try:
                dhd = float(dh)
                dv = float(ego) - float(lv)
                if dhd > 1e-3 and dv > 1e-2:
                    tc = dhd / dv
                    if math.isfinite(tc) and tc > 0.0:
                        tset = min(tc, 999.0)
            except (TypeError, ValueError):
                tset = 999.0
        r["ttc"] = "{:.6f}".format(tset)

    if _need("inv_ttc"):
        tt = _pf("ttc")
        if tt is not None:
            r["inv_ttc"] = "{:.9f}".format(reciprocal_inv_feature(tt))

    if _need("inv_time_headway"):
        th = _pf("time_headway")
        if th is not None:
            r["inv_time_headway"] = "{:.9f}".format(reciprocal_inv_feature(th))

    return r


def features_at_timestep(rows, timestamps, idx, features):
    """
    One training-aligned feature vector at row ``idx`` (used by train_bc_gru and generators).

    Returns list of floats, or ``None`` if any required scalar is missing/invalid.

    If ``features`` omits ``dt_prev`` (uniform sim grid), timestamps are unused for features.

    Args:
        rows: list of row dicts (from CSV).
        timestamps: list of floats/None aligned with rows (for ``dt_prev`` only).
        idx: row index ``0 .. len(rows)-1``.
        features: ordered feature names, same list as training.
    """
    r = rows[idx]
    if "dt_prev" not in features:
        return scalar_features_for_row(r, 0.0, features)
    if idx == 0:
        dtv = 0.0
    else:
        t0 = timestamps[idx - 1]
        t1 = timestamps[idx]
        if t0 is None or t1 is None:
            return None
        dtv = max(0.0, t1 - t0)
    return scalar_features_for_row(r, dtv, features)


def scalar_features_for_row(row_dict, dt_prev_value, features):
    """Single-row BC-GRU features; ``dt_prev_value`` matches training (0 at first row)."""
    fv = []
    for k in features:
        if k == "dt_prev":
            v = float(dt_prev_value)
        else:
            v = _row_value(row_dict, k)
        if v is None:
            return None
        fv.append(v)
    return fv
