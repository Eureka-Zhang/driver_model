# -*- coding: utf-8 -*-
"""
Train a **residual** GRU for car-following with **Scheduled Sampling** closed-loop
training. IDM gives the longitudinal baseline; the GRU learns Δa = a_real - a_IDM
under trajectory rollouts, not isolated windows.

## Why scheduled sampling

The open-loop residual trainer (``train_bc_gru_residual.py``) suffers exposure
bias: training windows contain real ``(v, gap, a_prev, a_IDM_real)``, but at
inference the window is built from closed-loop ``(v_cf, gap_cf, a_cf, a_IDM_cf)``
where ``a_cf = a_IDM_cf + Δa_GRU``. The distributions drift apart after a few
tens of steps.

Scheduled sampling trains on **rolled-out trajectories**: for each step t the
GRU predicts Δa and we integrate; with probability ``p_teacher`` we snap state
back to the real log, with probability ``1 - p_teacher`` we keep the predicted
state. ``p_teacher`` anneals from 1.0 down to a floor over training so the
network sees the exact conditions it will face at deployment.

## Key differences vs open-loop trainer

| | Open-loop v1 | Scheduled-sampling v2 |
|---|---|---|
| Sample unit | (window, target) pair | entire CSV segment |
| Loss per batch | 1 prediction × batch | ~BPTT_window predictions × seg_batch |
| ``a_prev`` in window | always ``a_real_prev`` | mix of real/predicted per p_teacher |
| ``(v, gap)`` in window | real log values | rolled-out values at 1-p_teacher |
| Gradient path | 1 step | truncated BPTT every ``--bptt_window`` steps |

## Usage

    python3 following/gru_train/train_bc_gru_residual_v2.py \\
      --data_dir /home/zwx/driver_model/following/outputs/following_calibrated \\
      --idm_dir /home/zwx/driver_model/following/outputs/idm_per_driver \\
      --out_dir /home/zwx/driver_model/following/outputs/residual_gru_v2/T5 \\
      --drivers T5 \\
      --seq_len 20 --bptt_window 50 \\
      --epochs 60 --teacher_floor 0.3 --teacher_anneal_epochs 25 \\
      --min_sim_time_s 15
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import re
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from bc_gru_features import (
    hydrate_bc_gru_row_aliases,
    reciprocal_inv_feature,
    _parse_float,
    _row_value,
)


# ------------------------------------------------------------------
# IDM helpers (same formula as fit_idm_per_driver.py)
# ------------------------------------------------------------------


def _idm_accel(v, vl, gap, params, eps=1e-3):
    v0 = float(params["v0"])
    s0 = float(params["s0"])
    a = float(params["a"])
    b = float(params["b"])
    T = float(params["T"])
    delta = float(params.get("delta", 4.0))
    s = max(gap, eps)
    dv = v - vl
    sab = math.sqrt(max(a * b, 1e-12))
    s_star = s0 + v * T + v * dv / (2.0 * sab)
    s_star = max(s_star, s0 + 1e-4)
    v0_safe = max(v0, eps)
    ratio = min(v / v0_safe, 50.0) if v0_safe > 0 else 0.0
    free_term = ratio ** delta
    return a * (1.0 - free_term - (s_star / s) ** 2)


def _idm_accel_torch(v, vl, gap, p, eps=1e-3):
    """Differentiable IDM in torch (batched scalars)."""
    import torch
    s = torch.clamp(gap, min=eps)
    dv = v - vl
    sab = math.sqrt(max(float(p["a"]) * float(p["b"]), 1e-12))
    s_star = float(p["s0"]) + v * float(p["T"]) + v * dv / (2.0 * sab)
    s_star = torch.clamp(s_star, min=float(p["s0"]) + 1e-4)
    ratio = torch.clamp(v / max(float(p["v0"]), eps), max=50.0)
    free_term = torch.pow(torch.clamp(ratio, min=0.0), float(p.get("delta", 4.0)))
    acc = float(p["a"]) * (1.0 - free_term - torch.pow(s_star / s, 2))
    return acc


def _load_driver_idm(idm_dir, driver_id):
    fp = os.path.join(idm_dir, driver_id, "idm.json")
    if not os.path.isfile(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f).get("parameters")


# ------------------------------------------------------------------
# Segment loading
# ------------------------------------------------------------------


def _discover_following_csvs(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn == "driving_data.csv" or re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _load_segment(csv_path, time_column, time_fallback_column):
    """Return dict of per-row arrays for an entire CSV (sorted by time)."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    rows = [hydrate_bc_gru_row_aliases(r) for r in rows]
    if not rows:
        return None
    ts = [_parse_float(r.get(time_column)) for r in rows]
    if any(t is None for t in ts):
        ts = [_parse_float(r.get(time_fallback_column)) for r in rows]
        if any(t is None for t in ts):
            return None
    order = sorted(range(len(rows)), key=lambda i: ts[i])
    rows = [rows[i] for i in order]
    ts = [ts[i] for i in order]

    n = len(rows)
    v = np.zeros(n, dtype=np.float32)
    a = np.zeros(n, dtype=np.float32)
    gap = np.zeros(n, dtype=np.float32)
    v_lead = np.zeros(n, dtype=np.float32)
    rel_v = np.zeros(n, dtype=np.float32)
    inv_ttc = np.zeros(n, dtype=np.float32)
    inv_thw = np.zeros(n, dtype=np.float32)
    ok = np.ones(n, dtype=bool)
    for i, r in enumerate(rows):
        ev = _row_value(r, "ego_v_long")
        ea = _row_value(r, "ego_a_long")
        gp = _row_value(r, "distance_headway")
        rv = _row_value(r, "relative_v_long")
        lv = _row_value(r, "lead_v_long")
        if ev is None or ea is None or gp is None or rv is None or lv is None:
            ok[i] = False
            continue
        v[i], a[i], gap[i] = float(ev), float(ea), float(gp)
        rel_v[i], v_lead[i] = float(rv), float(lv)
        inv_ttc[i] = reciprocal_inv_feature(_parse_float(r.get("ttc")))
        inv_thw[i] = reciprocal_inv_feature(_parse_float(r.get("time_headway")))

    return dict(
        v=v, a=a, gap=gap, v_lead=v_lead, rel_v=rel_v,
        inv_ttc=inv_ttc, inv_thw=inv_thw,
        ts=np.asarray(ts, dtype=np.float64), ok=ok,
        driver_id=_extract_driver_id(csv_path),
        path=csv_path,
    )


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--idm_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--drivers", type=str, default="",
                    help="Comma-separated driver IDs; empty = all found.")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--bptt_window", type=int, default=50,
                    help="Truncated BPTT: backprop every N rolled-out steps.")
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seg_batch", type=int, default=4,
                    help="Number of segments processed per batch (rolled in parallel).")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--residual_clip", type=float, default=4.0)
    ap.add_argument("--accel_clip_min", type=float, default=-8.0)
    ap.add_argument("--accel_clip_max", type=float, default=6.0)
    ap.add_argument("--lambda_v", type=float, default=0.1)
    ap.add_argument("--lambda_gap", type=float, default=0.05)
    ap.add_argument("--lambda_residual_l2", type=float, default=0.01)
    ap.add_argument("--teacher_floor", type=float, default=0.3,
                    help="Lower bound on teacher probability after annealing.")
    ap.add_argument("--teacher_anneal_epochs", type=int, default=25,
                    help="Epochs over which teacher probability anneals from 1.0 to floor.")
    ap.add_argument("--teacher_warmup_epochs", type=int, default=3,
                    help="Epochs of pure teacher forcing before annealing starts.")
    ap.add_argument("--min_sim_time_s", type=float, default=15.0)
    ap.add_argument("--time_column", type=str, default="sim_time_s")
    ap.add_argument("--time_fallback_column", type=str, default="timestamp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    # --- Discover CSVs per driver ---
    all_paths = _discover_following_csvs(args.data_dir)
    paths_by_driver = {}
    for p in all_paths:
        paths_by_driver.setdefault(_extract_driver_id(p), []).append(p)

    if args.drivers:
        wanted = [d.strip() for d in args.drivers.split(",") if d.strip()]
        paths_by_driver = {d: paths_by_driver[d] for d in wanted if d in paths_by_driver}
    if not paths_by_driver:
        raise SystemExit("[ERR] no drivers matched")

    # --- Per-driver split (within-driver by ratio) ---
    rng = random.Random(args.seed)
    train_segs, val_segs, test_segs = [], [], []
    idm_by_driver = {}
    for d, paths in paths_by_driver.items():
        idm = _load_driver_idm(args.idm_dir, d)
        if idm is None:
            print("[SKIP] {}: no idm.json".format(d))
            continue
        idm_by_driver[d] = idm
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_tr = max(1, int(round(n * args.train_ratio)))
        n_va = max(1, int(round(n * args.val_ratio))) if n > 2 else max(0, n - n_tr)
        train_p = paths[:n_tr]
        val_p = paths[n_tr:n_tr + n_va]
        test_p = paths[n_tr + n_va:]
        for p in train_p:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None:
                train_segs.append(s)
        for p in val_p:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None:
                val_segs.append(s)
        for p in test_p:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None:
                test_segs.append(s)
        print("[{}] train={} val={} test={}  idm=v0:{:.2f} s0:{:.2f} a:{:.2f} b:{:.2f} T:{:.2f}".format(
            d, len(train_p), len(val_p), len(test_p),
            idm["v0"], idm["s0"], idm["a"], idm["b"], idm["T"]))

    if not train_segs:
        raise SystemExit("[ERR] empty train set")

    # --- Feature normalization from train set, teacher-forced reality ---
    def _build_feat_matrix(segs):
        feats = []
        for s in segs:
            a_idm = np.zeros_like(s["v"])
            for i in range(len(s["v"])):
                a_idm[i] = _idm_accel(
                    float(s["v"][i]), float(s["v_lead"][i]),
                    max(float(s["gap"][i]), 0.5),
                    idm_by_driver[s["driver_id"]],
                )
            s["a_idm"] = a_idm.astype(np.float32)
            # Build 8-D feature: [v, a, gap, rel_v, v_lead, inv_ttc, inv_thw, a_idm]
            f = np.stack([
                s["v"], s["a"], s["gap"], s["rel_v"], s["v_lead"],
                s["inv_ttc"], s["inv_thw"], s["a_idm"],
            ], axis=1)
            feats.append(f)
        return np.concatenate(feats, axis=0)

    feat_all_train = _build_feat_matrix(train_segs)
    # Also compute a_idm for val/test so their a_idm column exists.
    for s in val_segs + test_segs:
        a_idm = np.zeros_like(s["v"])
        for i in range(len(s["v"])):
            a_idm[i] = _idm_accel(
                float(s["v"][i]), float(s["v_lead"][i]),
                max(float(s["gap"][i]), 0.5),
                idm_by_driver[s["driver_id"]],
            )
        s["a_idm"] = a_idm.astype(np.float32)
    feat_mean = feat_all_train.mean(axis=0)
    feat_std = feat_all_train.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    print("[norm] feature_mean={}\n[norm] feature_std={}".format(
        np.round(feat_mean, 4), np.round(feat_std, 4)))

    FEATURE_NAMES = [
        "ego_v_long", "ego_a_long", "distance_headway", "relative_v_long",
        "lead_v_long", "inv_ttc", "inv_time_headway", "a_idm",
    ]

    # --- Model ---
    class ResidualGRU(nn.Module):
        def __init__(self, d_in, d_hid, n_layers, dropout):
            super(ResidualGRU, self).__init__()
            self.gru = nn.GRU(
                input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
                batch_first=True, dropout=(dropout if n_layers > 1 else 0.0),
            )
            self.head = nn.Sequential(
                nn.Linear(d_hid, d_hid // 2),
                nn.ReLU(),
                nn.Linear(d_hid // 2, 1),
            )

        def forward(self, x, h=None):
            out, h = self.gru(x, h)
            return self.head(out[:, -1, :]), h

    model = ResidualGRU(len(FEATURE_NAMES), args.hidden_size, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mean_t = torch.from_numpy(feat_mean).float().to(device)
    std_t = torch.from_numpy(feat_std).float().to(device)

    def _teacher_prob(ep):
        if ep <= args.teacher_warmup_epochs:
            return 1.0
        if args.teacher_anneal_epochs <= 0:
            return args.teacher_floor
        progress = (ep - args.teacher_warmup_epochs) / float(args.teacher_anneal_epochs)
        progress = min(1.0, max(0.0, progress))
        return 1.0 - (1.0 - args.teacher_floor) * progress

    # ---------------------------------------------------------------
    # Rollout one segment with scheduled sampling; returns scalar loss
    # (divided by rollout length) and diagnostics.
    # ---------------------------------------------------------------
    def _rollout_segment(seg, p_teacher, train, collect_traj=False):
        idm = idm_by_driver[seg["driver_id"]]
        v_real = torch.from_numpy(seg["v"]).float().to(device)
        a_real = torch.from_numpy(seg["a"]).float().to(device)
        gap_real = torch.from_numpy(seg["gap"]).float().to(device)
        vlead = torch.from_numpy(seg["v_lead"]).float().to(device)
        rel_v_real = torch.from_numpy(seg["rel_v"]).float().to(device)
        inv_ttc_real = torch.from_numpy(seg["inv_ttc"]).float().to(device)
        inv_thw_real = torch.from_numpy(seg["inv_thw"]).float().to(device)
        ts = seg["ts"]
        n = len(v_real)

        dt_arr = np.diff(ts).astype(np.float32)
        dt_arr = np.concatenate([[0.05], dt_arr])
        dt_arr = np.clip(dt_arr, 0.01, 0.2)
        dt_t = torch.from_numpy(dt_arr).float().to(device)

        # Find first index where sim_time >= min_sim_time_s AND k >= seq_len
        first_pred = args.seq_len
        if args.min_sim_time_s > 0:
            cut = int(np.searchsorted(ts, args.min_sim_time_s))
            first_pred = max(first_pred, cut)
        if first_pred >= n - 1:
            return None

        # Initialize closed-loop state from the real state at (first_pred - 1)
        v_cur = v_real[first_pred - 1].clone()
        gap_cur = gap_real[first_pred - 1].clone()
        a_prev = a_real[first_pred - 1].clone()  # last observed accel goes into window

        # History window buffer: past ``seq_len`` rows as they were fed forward.
        # We build it from the real log up to first_pred (pure teacher forcing).
        hist_v = v_real[first_pred - args.seq_len:first_pred].clone()
        hist_gap = gap_real[first_pred - args.seq_len:first_pred].clone()
        hist_a = a_real[first_pred - args.seq_len:first_pred].clone()
        hist_vlead = vlead[first_pred - args.seq_len:first_pred].clone()
        hist_relv = rel_v_real[first_pred - args.seq_len:first_pred].clone()
        hist_invttc = inv_ttc_real[first_pred - args.seq_len:first_pred].clone()
        hist_invthw = inv_thw_real[first_pred - args.seq_len:first_pred].clone()
        # a_idm history computed from the same (real) states:
        hist_aidm = torch.from_numpy(seg["a_idm"][first_pred - args.seq_len:first_pred]).float().to(device)

        total_loss = torch.zeros((), device=device)
        count = 0
        running_loss = torch.zeros((), device=device)
        running_count = 0

        diag_res = 0.0
        diag_vmse = 0.0
        diag_gapmse = 0.0
        diag_n = 0

        h = None  # GRU hidden; detached every BPTT window

        for t in range(first_pred, n):
            # --- Build the current window tensor (seq_len, 8) ---
            a_idm_now = _idm_accel_torch(v_cur, vlead[t], torch.clamp(gap_cur, min=0.5), idm)

            window = torch.stack([
                hist_v, hist_a, hist_gap, hist_relv, hist_vlead,
                hist_invttc, hist_invthw, hist_aidm,
            ], dim=1).unsqueeze(0)  # (1, seq_len, 8)
            window_norm = (window - mean_t) / std_t

            delta_pred, h = model(window_norm, h)
            delta_clipped = torch.clamp(delta_pred.squeeze(), -args.residual_clip, args.residual_clip)

            a_pred = a_idm_now + delta_clipped
            a_pred = torch.clamp(a_pred, args.accel_clip_min, args.accel_clip_max)

            # --- Losses vs real ---
            a_t_real = a_real[t]
            residual_target = a_t_real - a_idm_now.detach()  # real residual
            loss_res = (delta_clipped - residual_target) ** 2

            # Predicted next step (for consistency)
            dt = dt_t[t]
            v_next_pred = torch.clamp(v_cur + a_pred * dt, min=0.0)
            gap_next_pred = gap_cur + (vlead[t] - v_cur) * dt
            v_next_real = v_real[t]
            gap_next_real = gap_real[t]
            loss_v = (v_next_pred - v_next_real) ** 2
            loss_g = (gap_next_pred - gap_next_real) ** 2
            loss_l2 = delta_clipped ** 2

            step_loss = (loss_res
                         + args.lambda_v * loss_v
                         + args.lambda_gap * loss_g
                         + args.lambda_residual_l2 * loss_l2)

            running_loss = running_loss + step_loss
            running_count += 1
            total_loss = total_loss + step_loss.detach()
            count += 1
            diag_res += float(loss_res.detach())
            diag_vmse += float(loss_v.detach())
            diag_gapmse += float(loss_g.detach())
            diag_n += 1

            # --- Truncated BPTT ---
            if train and running_count >= args.bptt_window:
                opt.zero_grad()
                running_mean = running_loss / float(running_count)
                running_mean.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                running_loss = torch.zeros((), device=device)
                running_count = 0
                # Detach state and history to cut the graph at this boundary
                v_cur = v_cur.detach()
                gap_cur = gap_cur.detach()
                a_prev = a_prev.detach()
                hist_v = hist_v.detach()
                hist_gap = hist_gap.detach()
                hist_a = hist_a.detach()
                hist_aidm = hist_aidm.detach()
                if h is not None:
                    h = h.detach()

            # --- Scheduled sampling: decide next state ---
            use_teacher = random.random() < p_teacher
            if use_teacher:
                new_v = v_real[t]
                new_gap = gap_real[t]
                new_a_prev = a_real[t]
                new_aidm_prev = torch.from_numpy(seg["a_idm"][t:t + 1]).float().to(device).squeeze()
                new_relv = rel_v_real[t]
                new_invttc = inv_ttc_real[t]
                new_invthw = inv_thw_real[t]
            else:
                new_v = v_next_pred
                new_gap = torch.clamp(gap_next_pred, min=0.1)
                new_a_prev = a_pred
                new_aidm_prev = a_idm_now
                new_relv = vlead[t] - new_v
                # closed-loop ttc/thw
                close_rate = new_v - vlead[t]
                new_invttc = torch.where(
                    close_rate > 0.01,
                    close_rate / torch.clamp(new_gap, min=0.5),
                    torch.zeros_like(close_rate),
                )
                new_invthw = torch.where(
                    new_v > 0.1,
                    new_v / torch.clamp(new_gap, min=0.5),
                    torch.zeros_like(new_v),
                )

            # --- Advance the rolling window (drop oldest, push new at end) ---
            hist_v = torch.cat([hist_v[1:], new_v.unsqueeze(0)])
            hist_gap = torch.cat([hist_gap[1:], new_gap.unsqueeze(0)])
            hist_a = torch.cat([hist_a[1:], new_a_prev.unsqueeze(0)])
            hist_aidm = torch.cat([hist_aidm[1:], new_aidm_prev.unsqueeze(0)])
            hist_vlead = torch.cat([hist_vlead[1:], vlead[t:t + 1]])
            hist_relv = torch.cat([hist_relv[1:], new_relv.unsqueeze(0)])
            hist_invttc = torch.cat([hist_invttc[1:], new_invttc.unsqueeze(0)])
            hist_invthw = torch.cat([hist_invthw[1:], new_invthw.unsqueeze(0)])

            v_cur = new_v
            gap_cur = new_gap
            a_prev = new_a_prev

        # Final BPTT step for remaining accumulated loss
        if train and running_count > 0:
            opt.zero_grad()
            running_mean = running_loss / float(running_count)
            running_mean.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        if diag_n == 0:
            return None
        return (
            float(total_loss.detach()) / max(1, count),
            diag_res / diag_n,
            diag_vmse / diag_n,
            diag_gapmse / diag_n,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_model.pt")
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history = []

    for ep in range(1, args.epochs + 1):
        p = _teacher_prob(ep)
        rng.shuffle(train_segs)

        # --- Train epoch ---
        model.train(True)
        tr_loss = tr_res = tr_v = tr_g = 0.0
        tr_n = 0
        for seg in train_segs:
            r = _rollout_segment(seg, p, train=True)
            if r is None:
                continue
            tr_loss += r[0]; tr_res += r[1]; tr_v += r[2]; tr_g += r[3]
            tr_n += 1
        if tr_n == 0:
            print("[WARN] epoch {}: no train rollouts".format(ep))
            continue
        tr_loss /= tr_n; tr_res /= tr_n; tr_v /= tr_n; tr_g /= tr_n

        # --- Validation: force fully closed-loop (p=0) ---
        model.train(False)
        va_loss = va_res = va_v = va_g = 0.0
        va_n = 0
        with __import__("torch").no_grad():
            for seg in val_segs:
                r = _rollout_segment(seg, 0.0, train=False)
                if r is None:
                    continue
                va_loss += r[0]; va_res += r[1]; va_v += r[2]; va_g += r[3]
                va_n += 1
        if va_n == 0:
            print("[WARN] epoch {}: no val rollouts".format(ep))
            continue
        va_loss /= va_n; va_res /= va_n; va_v /= va_n; va_g /= va_n

        history.append(dict(
            epoch=ep, p_teacher=p,
            train_loss=tr_loss, train_res=tr_res, train_v=tr_v, train_gap=tr_g,
            val_loss=va_loss, val_res=va_res, val_v=va_v, val_gap=va_g,
        ))
        print("ep {:3d}  p_teacher={:.2f}  train[res={:.4f} v={:.4f} gap={:.4f}]  "
              "val[res={:.4f} v={:.4f} gap={:.4f}]".format(
                  ep, p, tr_res, tr_v, tr_g, va_res, va_v, va_g))

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = ep
            bad = 0
            import torch as _t
            _t.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping at epoch {}".format(ep))
                break

    # --- Test: fully closed-loop ---
    test_metrics = None
    if test_segs and os.path.isfile(best_path):
        import torch as _t
        model.load_state_dict(_t.load(best_path, map_location=device))
        model.train(False)
        te_loss = te_res = te_v = te_g = 0.0
        te_n = 0
        with _t.no_grad():
            for seg in test_segs:
                r = _rollout_segment(seg, 0.0, train=False)
                if r is None:
                    continue
                te_loss += r[0]; te_res += r[1]; te_v += r[2]; te_g += r[3]
                te_n += 1
        if te_n > 0:
            test_metrics = dict(
                loss=te_loss / te_n, residual_mse=te_res / te_n,
                v_mse=te_v / te_n, gap_mse=te_g / te_n, n=te_n,
            )
            print("TEST (closed-loop)  res={:.4f}  v={:.4f}  gap={:.4f}  n={}".format(
                te_res / te_n, te_v / te_n, te_g / te_n, te_n))

    meta = dict(
        arch="residual_gru_scheduled_sampling",
        features=FEATURE_NAMES,
        targets=["delta_a"],
        target_mode="residual_delta_a",
        seq_len=args.seq_len,
        bptt_window=args.bptt_window,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_dim=len(FEATURE_NAMES),
        idm_dir=os.path.abspath(args.idm_dir),
        residual_clip=args.residual_clip,
        accel_clip=[args.accel_clip_min, args.accel_clip_max],
        lambda_v=args.lambda_v,
        lambda_gap=args.lambda_gap,
        lambda_residual_l2=args.lambda_residual_l2,
        teacher_floor=args.teacher_floor,
        teacher_anneal_epochs=args.teacher_anneal_epochs,
        teacher_warmup_epochs=args.teacher_warmup_epochs,
        min_sim_time_s=args.min_sim_time_s,
        drivers=list(paths_by_driver.keys()),
    )
    with open(os.path.join(args.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = dict(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        test=test_metrics,
        feature_mean=feat_mean.tolist(),
        feature_std=feat_std.tolist(),
        history=history,
    )
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[OK] saved {} (best epoch {})".format(best_path, best_epoch))


if __name__ == "__main__":
    main()
