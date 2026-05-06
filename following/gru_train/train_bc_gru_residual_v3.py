# -*- coding: utf-8 -*-
"""
Train a **gain+residual GRU** policy for car-following with short-horizon
closed-loop rollouts and **state anchoring** to keep training features aligned
with the teacher trajectory.

## Architecture

    a_pred = (1 + alpha) * a_IDM(state_cf) + delta_a

The GRU outputs ``(alpha_raw, delta_raw)``; squashing gives

    alpha   = 2.5 * sigmoid(alpha_raw) - 0.5   in (-0.5, 2.0)
    delta_a = 2.0 * tanh(delta_raw)            in (-2.0, 2.0)

``alpha`` rescales the IDM gain per-step (e.g. alpha=2 triples the IDM response
so calibrated conservative IDMs can express aggressive driver styles),
``delta_a`` adds fine-grained residual shaping.

## Why short segments + anchoring

Long closed-loop rollouts (v2) cause state_cf to drift from state_real, so
``a_IDM(state_cf)`` no longer matches the IDM the teacher experienced, and
``a_real`` loses its semantic tie to the network's input window.

Solution: split each CSV into overlapping sub-segments (default 25 frames),
initialize each sub-segment from the **real** state, roll out closed-loop but
softly pull ``(v, gap)`` back toward the real log every step:

    v_cf = (1 - beta) * v_free + beta * v_real
    gap_cf = (1 - beta) * gap_free + beta * gap_real

``beta`` anneals from 0.5 (strong anchor, early epochs) to 0.1 (weak anchor,
late epochs). Combined with the short horizon, state drift stays small so the
window features at every step remain close to what the teacher saw.

At inference time no anchoring is applied; the GRU has learned to output
accelerations that implicitly minimize drift because that was the training
signal.

## Usage

    python3 following/gru_train/train_bc_gru_residual_v3.py \\
      --data_dir /home/zwx/driver_model/following/outputs/following_calibrated \\
      --idm_dir /home/zwx/driver_model/following/outputs/idm_per_driver \\
      --out_dir /home/zwx/driver_model/following/outputs/residual_gru_v3/T5 \\
      --drivers T5 \\
      --seq_len 20 --segment_len 25 --segment_stride 15 \\
      --epochs 60 --anchor_start 0.5 --anchor_floor 0.1 --anchor_anneal_epochs 25 \\
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


FEATURE_NAMES = [
    "ego_v_long",
    "ego_a_long",
    "distance_headway",
    "relative_v_long",
    "lead_v_long",
    "inv_ttc",
    "inv_time_headway",
]


def _idm_accel(v, vl, gap, params, eps=1e-3):
    v0, s0 = float(params["v0"]), float(params["s0"])
    a, b, T = float(params["a"]), float(params["b"]), float(params["T"])
    delta = float(params.get("delta", 4.0))
    s = max(gap, eps)
    dv = v - vl
    sab = math.sqrt(max(a * b, 1e-12))
    s_star = max(s0 + v * T + v * dv / (2.0 * sab), s0 + 1e-4)
    v0_safe = max(v0, eps)
    ratio = min(v / v0_safe, 50.0) if v0_safe > 0 else 0.0
    return a * (1.0 - ratio ** delta - (s_star / s) ** 2)


def _idm_accel_torch(v, vl, gap, p, eps=1e-3):
    import torch
    s = torch.clamp(gap, min=eps)
    dv = v - vl
    sab = math.sqrt(max(float(p["a"]) * float(p["b"]), 1e-12))
    s_star = torch.clamp(
        float(p["s0"]) + v * float(p["T"]) + v * dv / (2.0 * sab),
        min=float(p["s0"]) + 1e-4,
    )
    ratio = torch.clamp(v / max(float(p["v0"]), eps), min=0.0, max=50.0)
    free_term = torch.pow(ratio, float(p.get("delta", 4.0)))
    return float(p["a"]) * (1.0 - free_term - torch.pow(s_star / s, 2))


def _load_driver_idm(idm_dir, driver_id):
    fp = os.path.join(idm_dir, driver_id, "idm.json")
    if not os.path.isfile(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f).get("parameters")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--idm_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--drivers", type=str, default="")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--split_mode", type=str, default="per_file",
                    choices=["per_file", "per_subsegment"],
                    help="per_file: split CSVs into train/val/test (default). "
                         "per_subsegment: put all CSVs into train AND val pool and split "
                         "their sub-segment lists temporally (first train_ratio fraction -> train, "
                         "next val_ratio -> val, rest -> test). Useful when a driver has very few CSVs.")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--segment_len", type=int, default=25,
                    help="Frames per training sub-segment (short horizon rollout).")
    ap.add_argument("--segment_stride", type=int, default=15,
                    help="Stride between sub-segments within one CSV (smaller = more samples).")
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seg_batch", type=int, default=32,
                    help="Sub-segments per gradient update.")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--alpha_lo", type=float, default=-0.5,
                    help="Lower bound for IDM gain correction alpha.")
    ap.add_argument("--alpha_hi", type=float, default=2.0,
                    help="Upper bound for IDM gain correction alpha.")
    ap.add_argument("--delta_clip", type=float, default=2.0,
                    help="Max |delta_a| after tanh.")
    ap.add_argument("--accel_clip_min", type=float, default=-8.0)
    ap.add_argument("--accel_clip_max", type=float, default=6.0)
    ap.add_argument("--lambda_v", type=float, default=0.1)
    ap.add_argument("--lambda_gap", type=float, default=0.05)
    ap.add_argument("--lambda_alpha", type=float, default=0.05,
                    help="Regularizer on alpha^2 (prefer IDM when it is good).")
    ap.add_argument("--lambda_delta", type=float, default=0.02,
                    help="Regularizer on delta_a^2 (keep residuals small).")
    ap.add_argument("--anchor_start", type=float, default=0.5,
                    help="Initial anchoring coefficient beta (pull state_cf toward state_real).")
    ap.add_argument("--anchor_floor", type=float, default=0.1,
                    help="Final anchoring coefficient after annealing.")
    ap.add_argument("--anchor_anneal_epochs", type=int, default=25)
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

    all_paths = _discover_following_csvs(args.data_dir)
    paths_by_driver = {}
    for p in all_paths:
        paths_by_driver.setdefault(_extract_driver_id(p), []).append(p)
    if args.drivers:
        wanted = [d.strip() for d in args.drivers.split(",") if d.strip()]
        paths_by_driver = {d: paths_by_driver[d] for d in wanted if d in paths_by_driver}
    if not paths_by_driver:
        raise SystemExit("[ERR] no drivers matched")

    rng = random.Random(args.seed)
    # In per_file mode each CSV becomes either a train, val, or test "seg".
    # In per_subsegment mode all CSVs go to a single pool and later we carve
    # sub-segment tasks into train/val/test slices per CSV.
    train_segs, val_segs, test_segs = [], [], []
    all_segs_for_subsplit = []  # only used in per_subsegment mode
    idm_by_driver = {}
    for d, paths in paths_by_driver.items():
        idm = _load_driver_idm(args.idm_dir, d)
        if idm is None:
            print("[SKIP] {}: no idm.json".format(d)); continue
        idm_by_driver[d] = idm
        paths = list(paths); rng.shuffle(paths)

        if args.split_mode == "per_subsegment":
            loaded = []
            for p in paths:
                s = _load_segment(p, args.time_column, args.time_fallback_column)
                if s is not None:
                    loaded.append(s)
                    all_segs_for_subsplit.append(s)
            print("[{}] mode=per_subsegment files={}  idm[a={:.2f} b={:.2f} T={:.2f} "
                  "s0={:.2f} v0={:.2f}]".format(
                      d, len(loaded),
                      idm["a"], idm["b"], idm["T"], idm["s0"], idm["v0"]))
            # Every CSV contributes to train, val, AND test sub-segment pools later.
            # For feature-norm we only want training rows, handled later.
            continue

        # per_file split
        n = len(paths)
        n_tr = max(1, int(round(n * args.train_ratio)))
        n_va = max(1, int(round(n * args.val_ratio))) if n > 2 else max(0, n - n_tr)
        for p in paths[:n_tr]:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None: train_segs.append(s)
        for p in paths[n_tr:n_tr + n_va]:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None: val_segs.append(s)
        for p in paths[n_tr + n_va:]:
            s = _load_segment(p, args.time_column, args.time_fallback_column)
            if s is not None: test_segs.append(s)
        print("[{}] train={} val={} test={}  idm[a={:.2f} b={:.2f} T={:.2f} s0={:.2f} v0={:.2f}]".format(
            d, sum(1 for p in paths[:n_tr]), sum(1 for p in paths[n_tr:n_tr + n_va]),
            len(paths) - n_tr - n_va,
            idm["a"], idm["b"], idm["T"], idm["s0"], idm["v0"]))

    if args.split_mode == "per_subsegment":
        # For feature normalization we treat every loaded CSV as "training data";
        # the split happens at sub-segment granularity below.
        train_segs = list(all_segs_for_subsplit)

    if not train_segs:
        raise SystemExit("[ERR] empty train set")

    # ---- Feature normalization using teacher-forced real states (train only) ----
    feats = []
    for s in train_segs:
        f = np.stack([s["v"], s["a"], s["gap"], s["rel_v"], s["v_lead"],
                      s["inv_ttc"], s["inv_thw"]], axis=1)
        feats.append(f)
    feat_cat = np.concatenate(feats, axis=0)
    feat_mean = feat_cat.mean(axis=0)
    feat_std = feat_cat.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    print("[norm] mean={}\n[norm] std={}".format(
        np.round(feat_mean, 4), np.round(feat_std, 4)))

    mean_t = torch.from_numpy(feat_mean).float().to(device)
    std_t = torch.from_numpy(feat_std).float().to(device)

    # ---- Model ----
    class GainResidualGRU(nn.Module):
        def __init__(self, d_in, d_hid, n_layers, dropout, alpha_lo, alpha_hi, delta_clip):
            super(GainResidualGRU, self).__init__()
            self.gru = nn.GRU(
                input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
                batch_first=True, dropout=(dropout if n_layers > 1 else 0.0),
            )
            self.head = nn.Sequential(
                nn.Linear(d_hid, d_hid // 2),
                nn.ReLU(),
                nn.Linear(d_hid // 2, 2),
            )
            self.alpha_lo = alpha_lo
            self.alpha_hi = alpha_hi
            self.delta_clip = delta_clip

        def forward(self, x, h=None):
            out, h = self.gru(x, h)
            raw = self.head(out[:, -1, :])  # (B, 2)
            alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(raw[:, 0:1])
            delta = self.delta_clip * torch.tanh(raw[:, 1:2])
            return alpha, delta, h

    model = GainResidualGRU(
        len(FEATURE_NAMES), args.hidden_size, args.num_layers, args.dropout,
        args.alpha_lo, args.alpha_hi, args.delta_clip,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def _anchor_beta(ep):
        if ep <= 1:
            return args.anchor_start
        if args.anchor_anneal_epochs <= 0:
            return args.anchor_floor
        progress = min(1.0, max(0.0, (ep - 1) / float(args.anchor_anneal_epochs)))
        return args.anchor_start + (args.anchor_floor - args.anchor_start) * progress

    # ---- Build sub-segment index list ----
    def _subsegment_starts(seg):
        ts = seg["ts"]
        n = len(seg["v"])
        first_usable = args.seq_len
        if args.min_sim_time_s > 0:
            cut = int(np.searchsorted(ts, args.min_sim_time_s))
            first_usable = max(first_usable, cut)
        starts = []
        t0 = first_usable
        while t0 + args.segment_len < n:
            starts.append(t0)
            t0 += args.segment_stride
        return starts

    def _run_subsegment(seg, t_start, beta, train):
        """Roll out [t_start, t_start+segment_len) with anchored closed-loop."""
        idm = idm_by_driver[seg["driver_id"]]
        v_real = torch.from_numpy(seg["v"]).float().to(device)
        a_real = torch.from_numpy(seg["a"]).float().to(device)
        gap_real = torch.from_numpy(seg["gap"]).float().to(device)
        vlead = torch.from_numpy(seg["v_lead"]).float().to(device)
        relv = torch.from_numpy(seg["rel_v"]).float().to(device)
        invttc = torch.from_numpy(seg["inv_ttc"]).float().to(device)
        invthw = torch.from_numpy(seg["inv_thw"]).float().to(device)
        ts = seg["ts"]
        dt_arr = np.clip(np.diff(ts).astype(np.float32), 0.01, 0.2)
        dt_t = torch.from_numpy(np.concatenate([[0.05], dt_arr])).float().to(device)

        t_end = t_start + args.segment_len
        n = len(v_real)
        if t_end >= n:
            return None

        # ---- Initialize closed-loop state from real at t_start-1 ----
        v_cur = v_real[t_start - 1].clone()
        gap_cur = gap_real[t_start - 1].clone()
        # History window is fully teacher (real) for the seq_len before t_start.
        hist_v = v_real[t_start - args.seq_len:t_start].clone()
        hist_a = a_real[t_start - args.seq_len:t_start].clone()
        hist_gap = gap_real[t_start - args.seq_len:t_start].clone()
        hist_relv = relv[t_start - args.seq_len:t_start].clone()
        hist_vlead = vlead[t_start - args.seq_len:t_start].clone()
        hist_invttc = invttc[t_start - args.seq_len:t_start].clone()
        hist_invthw = invthw[t_start - args.seq_len:t_start].clone()

        loss_sum = torch.zeros((), device=device)
        loss_count = 0

        diag_res = 0.0; diag_v = 0.0; diag_g = 0.0; diag_a = 0.0; diag_d = 0.0
        diag_n = 0

        h = None
        for t in range(t_start, t_end):
            window = torch.stack([
                hist_v, hist_a, hist_gap, hist_relv, hist_vlead, hist_invttc, hist_invthw,
            ], dim=1).unsqueeze(0)  # (1, seq_len, 7)
            window_norm = (window - mean_t) / std_t

            alpha, delta, h = model(window_norm, h)
            alpha = alpha.squeeze()
            delta = delta.squeeze()

            a_idm_now = _idm_accel_torch(
                v_cur, vlead[t], torch.clamp(gap_cur, min=0.5), idm,
            )
            a_pred = (1.0 + alpha) * a_idm_now + delta
            a_pred_clipped = torch.clamp(a_pred, args.accel_clip_min, args.accel_clip_max)

            # ---- Losses vs teacher ----
            loss_res = (a_pred - a_real[t]) ** 2

            dt = dt_t[t]
            v_free = torch.clamp(v_cur + a_pred_clipped * dt, min=0.0)
            gap_free = gap_cur + (vlead[t] - v_cur) * dt
            loss_v = (v_free - v_real[t]) ** 2
            loss_g = (gap_free - gap_real[t]) ** 2
            loss_a = alpha ** 2
            loss_d = delta ** 2

            step_loss = (
                loss_res
                + args.lambda_v * loss_v
                + args.lambda_gap * loss_g
                + args.lambda_alpha * loss_a
                + args.lambda_delta * loss_d
            )
            loss_sum = loss_sum + step_loss
            loss_count += 1
            diag_res += float(loss_res.detach())
            diag_v += float(loss_v.detach())
            diag_g += float(loss_g.detach())
            diag_a += float(alpha.detach() ** 2)
            diag_d += float(delta.detach() ** 2)
            diag_n += 1

            # ---- Anchored state update ----
            v_next = (1.0 - beta) * v_free + beta * v_real[t]
            gap_next = (1.0 - beta) * gap_free + beta * gap_real[t]

            # Advance history (push closed-loop values at tail)
            new_relv = vlead[t] - v_next
            gap_safe = torch.clamp(gap_next, min=0.5)
            close_rate = v_next - vlead[t]
            new_invttc = torch.where(
                close_rate > 0.01, close_rate / gap_safe, torch.zeros_like(close_rate),
            )
            new_invthw = torch.where(
                v_next > 0.1, v_next / gap_safe, torch.zeros_like(v_next),
            )
            hist_v = torch.cat([hist_v[1:], v_next.unsqueeze(0)])
            hist_a = torch.cat([hist_a[1:], a_pred_clipped.unsqueeze(0)])
            hist_gap = torch.cat([hist_gap[1:], gap_next.unsqueeze(0)])
            hist_relv = torch.cat([hist_relv[1:], new_relv.unsqueeze(0)])
            hist_vlead = torch.cat([hist_vlead[1:], vlead[t:t + 1]])
            hist_invttc = torch.cat([hist_invttc[1:], new_invttc.unsqueeze(0)])
            hist_invthw = torch.cat([hist_invthw[1:], new_invthw.unsqueeze(0)])

            v_cur = v_next
            gap_cur = gap_next

        if loss_count == 0:
            return None
        mean_loss = loss_sum / float(loss_count)
        return mean_loss, diag_res / diag_n, diag_v / diag_n, diag_g / diag_n, diag_a / diag_n, diag_d / diag_n

    os.makedirs(args.out_dir, exist_ok=True)
    best_path = os.path.join(args.out_dir, "best_model.pt")
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    history = []

    # Precompute sub-segment tasks
    train_tasks = []
    val_tasks = []
    test_tasks = []

    if args.split_mode == "per_subsegment":
        # Use every CSV for all three splits, carving its sub-segment list
        # temporally: first train_ratio -> train, next val_ratio -> val, rest -> test.
        # This avoids the "only 1 CSV per driver is unusable" failure mode.
        seen = set()
        unique_segs = []
        for s in train_segs:  # in this mode train_segs == all_segs_for_subsplit
            key = s["path"]
            if key in seen:
                continue
            seen.add(key)
            unique_segs.append(s)
        for s in unique_segs:
            starts = _subsegment_starts(s)  # already time-ordered
            n_sub = len(starts)
            if n_sub == 0:
                continue
            n_tr = max(1, int(round(n_sub * args.train_ratio)))
            n_va = max(1, int(round(n_sub * args.val_ratio))) if n_sub - n_tr > 0 else 0
            if n_tr + n_va > n_sub:
                n_va = max(0, n_sub - n_tr)
            n_te = n_sub - n_tr - n_va
            for t0 in starts[:n_tr]:
                train_tasks.append((s, t0))
            for t0 in starts[n_tr:n_tr + n_va]:
                val_tasks.append((s, t0))
            for t0 in starts[n_tr + n_va:n_tr + n_va + n_te]:
                test_tasks.append((s, t0))
            print("[sub-split] {} -> train={} val={} test={}".format(
                os.path.basename(os.path.dirname(s["path"])),
                n_tr, n_va, n_te))
    else:
        for s in train_segs:
            for t0 in _subsegment_starts(s):
                train_tasks.append((s, t0))
        for s in val_segs:
            for t0 in _subsegment_starts(s):
                val_tasks.append((s, t0))
        for s in test_segs:
            for t0 in _subsegment_starts(s):
                test_tasks.append((s, t0))

    print("[data] train_tasks={} val_tasks={} test_tasks={}".format(
        len(train_tasks), len(val_tasks), len(test_tasks)))

    for ep in range(1, args.epochs + 1):
        beta = _anchor_beta(ep)
        rng.shuffle(train_tasks)

        model.train(True)
        batch_losses = []
        tr_res = tr_v = tr_g = tr_a = tr_d = 0.0
        tr_n = 0
        opt.zero_grad()
        for seg, t0 in train_tasks:
            r = _run_subsegment(seg, t0, beta, train=True)
            if r is None:
                continue
            step_loss, res, vv, gg, aa, dd = r
            batch_losses.append(step_loss)
            tr_res += res; tr_v += vv; tr_g += gg; tr_a += aa; tr_d += dd; tr_n += 1
            if len(batch_losses) >= args.seg_batch:
                total = torch.stack(batch_losses).mean()
                total.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad()
                batch_losses = []
        if batch_losses:
            total = torch.stack(batch_losses).mean()
            total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad()
        if tr_n == 0:
            print("[WARN] epoch {}: no train tasks".format(ep)); continue
        tr_res /= tr_n; tr_v /= tr_n; tr_g /= tr_n; tr_a /= tr_n; tr_d /= tr_n

        # --- Validation: fully free (beta=0), closed-loop ---
        model.train(False)
        va_res = va_v = va_g = 0.0; va_n = 0
        import torch as _t
        with _t.no_grad():
            for seg, t0 in val_tasks:
                r = _run_subsegment(seg, t0, 0.0, train=False)
                if r is None:
                    continue
                _, res, vv, gg, _, _ = r
                va_res += res; va_v += vv; va_g += gg; va_n += 1
        if va_n == 0:
            print("[WARN] epoch {}: no val tasks".format(ep)); continue
        va_res /= va_n; va_v /= va_n; va_g /= va_n
        val_loss = va_res + args.lambda_v * va_v + args.lambda_gap * va_g

        history.append(dict(
            epoch=ep, beta=beta,
            train_res=tr_res, train_v=tr_v, train_gap=tr_g,
            train_alpha_sq=tr_a, train_delta_sq=tr_d,
            val_res=va_res, val_v=va_v, val_gap=va_g, val_loss=val_loss,
        ))
        print("ep {:3d}  beta={:.2f}  tr[res={:.4f} v={:.4f} gap={:.4f} a2={:.3f} d2={:.3f}]  "
              "va[res={:.4f} v={:.4f} gap={:.4f}]".format(
                  ep, beta, tr_res, tr_v, tr_g, tr_a, tr_d, va_res, va_v, va_g))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = ep
            bad = 0
            _t.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping at epoch {}".format(ep))
                break

    # --- Test closed-loop ---
    test_metrics = None
    if test_tasks and os.path.isfile(best_path):
        import torch as _t
        model.load_state_dict(_t.load(best_path, map_location=device))
        model.train(False)
        te_res = te_v = te_g = 0.0; te_n = 0
        with _t.no_grad():
            for seg, t0 in test_tasks:
                r = _run_subsegment(seg, t0, 0.0, train=False)
                if r is None: continue
                _, res, vv, gg, _, _ = r
                te_res += res; te_v += vv; te_g += gg; te_n += 1
        if te_n > 0:
            test_metrics = dict(residual_mse=te_res / te_n, v_mse=te_v / te_n,
                                gap_mse=te_g / te_n, n=te_n)
            print("TEST  res={:.4f}  v={:.4f}  gap={:.4f}  n={}".format(
                te_res / te_n, te_v / te_n, te_g / te_n, te_n))

    meta = dict(
        arch="gain_residual_gru_v3",
        features=FEATURE_NAMES,
        output_mode="alpha_delta_a",
        alpha_bounds=[args.alpha_lo, args.alpha_hi],
        delta_clip=args.delta_clip,
        seq_len=args.seq_len,
        segment_len=args.segment_len,
        segment_stride=args.segment_stride,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_dim=len(FEATURE_NAMES),
        idm_dir=os.path.abspath(args.idm_dir),
        accel_clip=[args.accel_clip_min, args.accel_clip_max],
        lambda_v=args.lambda_v,
        lambda_gap=args.lambda_gap,
        lambda_alpha=args.lambda_alpha,
        lambda_delta=args.lambda_delta,
        anchor_start=args.anchor_start,
        anchor_floor=args.anchor_floor,
        anchor_anneal_epochs=args.anchor_anneal_epochs,
        min_sim_time_s=args.min_sim_time_s,
        drivers=list(paths_by_driver.keys()),
    )
    with open(os.path.join(args.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = dict(
        best_epoch=best_epoch, best_val_loss=best_val, test=test_metrics,
        feature_mean=feat_mean.tolist(), feature_std=feat_std.tolist(),
        history=history,
    )
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[OK] saved {} (best epoch {})".format(best_path, best_epoch))


if __name__ == "__main__":
    main()
