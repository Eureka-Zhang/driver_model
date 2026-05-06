# -*- coding: utf-8 -*-
"""
Single-sample inference helper for the **gain+residual GRU v3** policy.

Loads ``<model_dir>/best_model.pt`` + ``model_meta.json`` and exposes
``ResidualGRUPolicy.step(state, history)`` which returns the next-step
longitudinal acceleration ``a_pred = (1 + alpha) * a_IDM + delta_a``
for use in closed-loop rollouts.

Typical use::

    from following.gru_train.residual_gru_policy import ResidualGRUPolicy

    policy = ResidualGRUPolicy(
        model_dir="/home/zwx/driver_model/following/outputs/residual_gru_v3/T5",
        device="cuda",  # or "cpu" / "auto"
    )

    # Build a rolling buffer with seq_len past rows (real or closed-loop)
    hist = policy.init_history(
        v_seq=[...], a_seq=[...], gap_seq=[...],
        lead_v_seq=[...], rel_v_seq=[...],
        inv_ttc_seq=[...], inv_thw_seq=[...],
    )

    for t in range(T):
        a_pred, info = policy.step(
            v_cur=v, gap_cur=gap, lead_v=lead_v[t],
            history=hist,
        )
        # integrate
        v_next = max(0.0, v + a_pred * dt)
        gap_next = gap + (lead_v[t] - v) * dt
        # update buffer for next call
        hist = policy.push_history(hist, v_next, a_pred, gap_next, lead_v[t])
        v, gap = v_next, gap_next
"""
from __future__ import print_function

import json
import math
import os

import numpy as np
import torch
import torch.nn as nn


class _GainResidualGRU(nn.Module):
    """Mirror of the model defined inside train_bc_gru_residual_v3.py."""

    def __init__(self, d_in, d_hid, n_layers, dropout, alpha_lo, alpha_hi, delta_clip):
        super(_GainResidualGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
            batch_first=True, dropout=(dropout if n_layers > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(d_hid, d_hid // 2),
            nn.ReLU(),
            nn.Linear(d_hid // 2, 2),
        )
        self.alpha_lo = float(alpha_lo)
        self.alpha_hi = float(alpha_hi)
        self.delta_clip = float(delta_clip)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        raw = self.head(out[:, -1, :])  # (B, 2)
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(raw[:, 0:1])
        delta = self.delta_clip * torch.tanh(raw[:, 1:2])
        return alpha, delta, h


def _idm_accel(v, vl, gap, params, eps=1e-3):
    v0 = float(params["v0"]); s0 = float(params["s0"])
    a = float(params["a"]); b = float(params["b"]); T = float(params["T"])
    delta = float(params.get("delta", 4.0))
    s = max(gap, eps)
    dv = v - vl
    sab = math.sqrt(max(a * b, 1e-12))
    s_star = max(s0 + v * T + v * dv / (2.0 * sab), s0 + 1e-4)
    v0_safe = max(v0, eps)
    ratio = min(v / v0_safe, 50.0) if v0_safe > 0 else 0.0
    return a * (1.0 - ratio ** delta - (s_star / s) ** 2)


class ResidualGRUPolicy(object):
    """Per-driver residual IDM+GRU policy for car-following closed-loop rollout.

    Parameters
    ----------
    model_dir : str
        Directory with ``best_model.pt``, ``model_meta.json``, ``train_report.json``.
    device : str
        ``"auto" | "cuda" | "cpu"``.
    idm_params : dict, optional
        Override IDM parameters. If ``None``, loaded from ``<idm_dir>/<driver>/idm.json``
        where ``<driver>`` is read from ``model_meta.json["drivers"][0]``.
    driver_id : str, optional
        Override driver id used for IDM lookup.
    """

    FEATURE_ORDER = [
        "ego_v_long",
        "ego_a_long",
        "distance_headway",
        "relative_v_long",
        "lead_v_long",
        "inv_ttc",
        "inv_time_headway",
    ]

    def __init__(self, model_dir, device="auto", idm_params=None, driver_id=None):
        self.model_dir = os.path.abspath(model_dir)
        meta_fp = os.path.join(self.model_dir, "model_meta.json")
        report_fp = os.path.join(self.model_dir, "train_report.json")
        wts_fp = os.path.join(self.model_dir, "best_model.pt")
        if not (os.path.isfile(meta_fp) and os.path.isfile(report_fp) and os.path.isfile(wts_fp)):
            raise FileNotFoundError(
                "ResidualGRUPolicy needs best_model.pt + model_meta.json + train_report.json "
                "in {}".format(self.model_dir))

        with open(meta_fp, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(report_fp, "r", encoding="utf-8") as f:
            self.report = json.load(f)

        # sanity checks
        assert self.meta.get("arch") == "gain_residual_gru_v3", \
            "unexpected arch: {}".format(self.meta.get("arch"))
        feats = self.meta.get("features")
        assert feats == self.FEATURE_ORDER, \
            "feature order mismatch: {} != {}".format(feats, self.FEATURE_ORDER)

        self.seq_len = int(self.meta["seq_len"])
        self.input_dim = int(self.meta["input_dim"])
        self.alpha_bounds = tuple(self.meta["alpha_bounds"])
        self.delta_clip = float(self.meta["delta_clip"])
        self.accel_clip = tuple(self.meta["accel_clip"])

        # IDM parameters
        if idm_params is None:
            drv_id = driver_id or self.meta["drivers"][0]
            idm_dir = self.meta.get("idm_dir", "")
            idm_fp = os.path.join(idm_dir, drv_id, "idm.json")
            with open(idm_fp, "r", encoding="utf-8") as f:
                self.idm_params = json.load(f)["parameters"]
            self.driver_id = drv_id
        else:
            self.idm_params = dict(idm_params)
            self.driver_id = driver_id or self.meta["drivers"][0]

        # Normalization stats (train-set mean/std)
        self.feat_mean = np.asarray(self.report["feature_mean"], dtype=np.float32)
        self.feat_std = np.asarray(self.report["feature_std"], dtype=np.float32)

        # Device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = _GainResidualGRU(
            d_in=self.input_dim,
            d_hid=int(self.meta["hidden_size"]),
            n_layers=int(self.meta["num_layers"]),
            dropout=float(self.meta["dropout"]),
            alpha_lo=self.alpha_bounds[0],
            alpha_hi=self.alpha_bounds[1],
            delta_clip=self.delta_clip,
        ).to(self.device)
        self.model.load_state_dict(torch.load(wts_fp, map_location=self.device))
        self.model.eval()

        self._mean_t = torch.from_numpy(self.feat_mean).float().to(self.device)
        self._std_t = torch.from_numpy(self.feat_std).float().to(self.device)

    # ------------------------------------------------------------------
    # History helpers (pure numpy to keep call sites simple)
    # ------------------------------------------------------------------

    def init_history(self, v_seq, a_seq, gap_seq, lead_v_seq, rel_v_seq,
                     inv_ttc_seq, inv_thw_seq):
        """Build the initial (seq_len, 7) rolling window from real past rows.

        All inputs are arrays of at least ``seq_len`` length; only the last
        ``seq_len`` entries are used."""
        L = self.seq_len
        arrs = [np.asarray(x, dtype=np.float32)[-L:] for x in (
            v_seq, a_seq, gap_seq, rel_v_seq, lead_v_seq, inv_ttc_seq, inv_thw_seq,
        )]
        for name, a in zip(self.FEATURE_ORDER, arrs):
            if a.shape[0] != L:
                raise ValueError(
                    "init_history: {} has length {}, need {}".format(name, a.shape[0], L))
        hist = np.stack(arrs, axis=1)  # (L, 7)
        return hist.astype(np.float32)

    def push_history(self, hist, v_new, a_new, gap_new, lead_v_new):
        """Drop oldest row, append a new one computed from closed-loop state.

        ``rel_v``, ``inv_ttc``, ``inv_thw`` are derived consistently with training.
        """
        gap_safe = max(gap_new, 0.5)
        rel_v = lead_v_new - v_new
        close_rate = v_new - lead_v_new
        inv_ttc = (close_rate / gap_safe) if close_rate > 0.01 else 0.0
        inv_thw = (v_new / gap_safe) if v_new > 0.1 else 0.0
        row = np.asarray([
            v_new, a_new, gap_new, rel_v, lead_v_new, inv_ttc, inv_thw,
        ], dtype=np.float32)
        return np.concatenate([hist[1:], row[None, :]], axis=0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, v_cur, gap_cur, lead_v, history, return_alpha_delta=False):
        """Compute next-step acceleration from current state and history window.

        Parameters
        ----------
        v_cur : float
            Ego speed at current step (m/s).
        gap_cur : float
            Distance to lead at current step (m).
        lead_v : float
            Lead speed at current step (m/s).
        history : np.ndarray
            (seq_len, 7) rolling window built via ``init_history`` / ``push_history``.

        Returns
        -------
        a_pred : float
            Clipped longitudinal acceleration (m/s^2) ready to integrate.
        info : dict
            ``alpha``, ``delta_a``, ``a_idm``, ``a_raw``.
        """
        h = np.asarray(history, dtype=np.float32)
        assert h.shape == (self.seq_len, self.input_dim), \
            "history shape {} != expected ({}, {})".format(h.shape, self.seq_len, self.input_dim)

        x = torch.from_numpy(h).float().to(self.device).unsqueeze(0)  # (1, L, 7)
        x_norm = (x - self._mean_t) / self._std_t

        alpha, delta, _ = self.model(x_norm)
        alpha = float(alpha.squeeze().item())
        delta = float(delta.squeeze().item())

        a_idm = _idm_accel(float(v_cur), float(lead_v),
                           max(float(gap_cur), 0.5), self.idm_params)
        a_raw = (1.0 + alpha) * a_idm + delta
        a_pred = float(max(self.accel_clip[0], min(self.accel_clip[1], a_raw)))

        if return_alpha_delta:
            return a_pred, dict(alpha=alpha, delta_a=delta, a_idm=a_idm, a_raw=a_raw)
        return a_pred, dict(alpha=alpha, delta_a=delta, a_idm=a_idm, a_raw=a_raw)


# ------------------------------------------------------------------
# Convenience: closed-loop rollout across a prerecorded lead trajectory
# ------------------------------------------------------------------


def rollout_against_lead(policy, init_state, lead_trajectory, dt=0.05):
    """Simulate a full car-following trajectory given a lead-vehicle log.

    Parameters
    ----------
    policy : ResidualGRUPolicy
    init_state : dict with keys
        ``v_seq``, ``a_seq``, ``gap_seq``, ``lead_v_seq``,
        ``rel_v_seq``, ``inv_ttc_seq``, ``inv_thw_seq`` (each length >= seq_len);
        plus ``v0``, ``gap0`` for the first step.
    lead_trajectory : list-like of floats
        Lead speeds (m/s) for steps 0..T-1.
    dt : float
        Sim step (s). Constant for calibrated sim grids.

    Returns
    -------
    dict with arrays ``t``, ``v``, ``a``, ``gap``, ``alpha``, ``delta_a``.
    """
    hist = policy.init_history(
        init_state["v_seq"], init_state["a_seq"], init_state["gap_seq"],
        init_state["lead_v_seq"], init_state["rel_v_seq"],
        init_state["inv_ttc_seq"], init_state["inv_thw_seq"],
    )
    v = float(init_state["v0"])
    gap = float(init_state["gap0"])

    T = len(lead_trajectory)
    out = dict(
        t=np.arange(T) * dt,
        v=np.zeros(T, dtype=np.float32),
        a=np.zeros(T, dtype=np.float32),
        gap=np.zeros(T, dtype=np.float32),
        alpha=np.zeros(T, dtype=np.float32),
        delta_a=np.zeros(T, dtype=np.float32),
        a_idm=np.zeros(T, dtype=np.float32),
    )
    for t in range(T):
        lead_v = float(lead_trajectory[t])
        a_pred, info = policy.step(v, gap, lead_v, hist, return_alpha_delta=True)
        out["v"][t] = v
        out["a"][t] = a_pred
        out["gap"][t] = gap
        out["alpha"][t] = info["alpha"]
        out["delta_a"][t] = info["delta_a"]
        out["a_idm"][t] = info["a_idm"]

        v_next = max(0.0, v + a_pred * dt)
        gap_next = gap + (lead_v - v) * dt
        hist = policy.push_history(hist, v_next, a_pred, gap_next, lead_v)
        v, gap = v_next, gap_next

    return out


if __name__ == "__main__":
    import argparse
    import csv
    ap = argparse.ArgumentParser(description="Closed-loop rollout of a residual GRU v3 policy.")
    ap.add_argument("--model_dir", required=True,
                    help="Directory with best_model.pt / model_meta.json / train_report.json")
    ap.add_argument("--driving_csv", required=True,
                    help="Source driving_data.csv to replay the lead trajectory from.")
    ap.add_argument("--out_csv", default="",
                    help="Optional output CSV with columns t,v,a,gap,alpha,delta_a,a_idm,lead_v.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--start_time_s", type=float, default=0.0,
                    help="Begin closed-loop from this sim_time_s (default 0 = just after the seq_len seed window).")
    args = ap.parse_args()

    policy = ResidualGRUPolicy(args.model_dir, device=args.device)

    with open(args.driving_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n <= policy.seq_len + 1:
        raise SystemExit("CSV too short: need > seq_len+1 rows")

    def _col(name):
        return np.asarray(
            [float(r.get(name, 0.0) or 0.0) for r in rows], dtype=np.float32,
        )

    v_arr = _col("ego_v_long")
    a_arr = _col("ego_a_long")
    gap_arr = _col("distance_headway")
    rel_v_arr = _col("relative_v_long")
    lead_v_arr = _col("lead_v_long")
    inv_ttc_arr = _col("inv_ttc")
    inv_thw_arr = _col("inv_time_headway")
    ts_arr = _col("sim_time_s")

    L = policy.seq_len
    # Find the earliest index whose sim_time_s >= start_time_s AND has >= L past rows
    start_idx = L
    if args.start_time_s > 0:
        import numpy as _np
        cut = int(_np.searchsorted(ts_arr, args.start_time_s))
        start_idx = max(L, cut)
    if start_idx >= n:
        raise SystemExit("start_time_s too large for CSV")

    init_state = dict(
        v_seq=v_arr[start_idx - L:start_idx],
        a_seq=a_arr[start_idx - L:start_idx],
        gap_seq=gap_arr[start_idx - L:start_idx],
        lead_v_seq=lead_v_arr[start_idx - L:start_idx],
        rel_v_seq=rel_v_arr[start_idx - L:start_idx],
        inv_ttc_seq=inv_ttc_arr[start_idx - L:start_idx],
        inv_thw_seq=inv_thw_arr[start_idx - L:start_idx],
        v0=float(v_arr[start_idx - 1]),
        gap0=float(gap_arr[start_idx - 1]),
    )
    out = rollout_against_lead(policy, init_state, lead_v_arr[start_idx:])

    # metrics vs ground truth (aligned slice)
    v_gt = v_arr[start_idx:]; a_gt = a_arr[start_idx:]; gap_gt = gap_arr[start_idx:]
    rmse_v = float(np.sqrt(np.mean((out["v"] - v_gt) ** 2)))
    rmse_a = float(np.sqrt(np.mean((out["a"] - a_gt) ** 2)))
    rmse_gap = float(np.sqrt(np.mean((out["gap"] - gap_gt) ** 2)))
    print("[rollout] driver={} n={} RMSE: v={:.3f} a={:.3f} gap={:.3f} | "
          "alpha[mean={:.3f} std={:.3f}] delta[mean={:.3f} std={:.3f}]".format(
              policy.driver_id, len(out["v"]),
              rmse_v, rmse_a, rmse_gap,
              float(out["alpha"].mean()), float(out["alpha"].std()),
              float(out["delta_a"].mean()), float(out["delta_a"].std())))

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "v", "a", "gap", "alpha", "delta_a", "a_idm", "lead_v",
                        "v_gt", "a_gt", "gap_gt"])
            for i in range(len(out["v"])):
                w.writerow([
                    float(out["t"][i]), float(out["v"][i]), float(out["a"][i]),
                    float(out["gap"][i]), float(out["alpha"][i]),
                    float(out["delta_a"][i]), float(out["a_idm"][i]),
                    float(lead_v_arr[start_idx + i]),
                    float(v_gt[i]), float(a_gt[i]), float(gap_gt[i]),
                ])
        print("[saved] {}".format(args.out_csv))
