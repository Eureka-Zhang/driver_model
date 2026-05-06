# -*- coding: utf-8 -*-
"""
Train a **residual** GRU for car-following: IDM gives the baseline acceleration,
the GRU learns the driver-specific deviation Δa = a_real - a_IDM.

## Architecture

At each timestep t:

    a_pred(t) = a_IDM(t; state) + Δa_pred(t; past window)

    where a_IDM is the classic Treiber IDM using per-driver parameters fitted by
    ``following/idm/fit_idm_per_driver.py`` and Δa is the GRU output.

## Training (teacher forcing, open-loop)

For each row t in a calibrated CSV:
  1. Compute ``a_IDM(t)`` from the **real** (v, v_lead, gap) at time t.
  2. Compute residual target ``Δa_target(t) = a_real(t) - a_IDM(t)``.
  3. Build causal feature window ``X[t] = features[t-seq_len : t]`` containing
     both kinematics and ``a_IDM`` (so the network knows the baseline).
  4. Loss = MSE(Δa_pred, Δa_target) + λ_v · speed consistency + λ_j · jerk penalty.

The residual target has much smaller variance than raw acceleration (typically
std ≈ 0.3-0.6 vs 1.4), so the network only learns the "personalization" signal
and the IDM baseline guarantees physical safety in closed loop.

## Closed-loop inference

In ``generate_following_outputs_residual.py`` (to be created):
  - IDM is computed from the closed-loop state at each step.
  - GRU input window includes a_IDM as a feature.
  - Final a_pred = clip(a_IDM + Δa_GRU, a_min, a_max), then integrate v and gap.
  - If GRU fails / drops out, IDM alone still produces safe rollouts.

## Usage

    python3 following/gru_train/train_bc_gru_residual.py \\
      --data_dir /home/zwx/driver_model/following/outputs/following_calibrated \\
      --idm_dir /home/zwx/driver_model/following/outputs/idm_per_driver \\
      --out_dir /home/zwx/driver_model/following/outputs/residual_gru_per_driver/T5 \\
      --train_drivers T5 --val_drivers T5 --test_drivers T5 \\
      --split_within_driver --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \\
      --seq_len 20 --epochs 60 --min_sim_time_s 15
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
    DEFAULT_FEATURES,
    hydrate_bc_gru_row_aliases,
    reciprocal_inv_feature,
    _parse_float,
    _row_value,
)


# Feature order for residual training. ``a_idm`` appears at index -1 so the
# network always knows the baseline acceleration it is correcting.
RESIDUAL_FEATURES = [
    "ego_v_long",
    "ego_a_long",
    "distance_headway",
    "relative_v_long",
    "lead_v_long",
    "inv_ttc",
    "inv_time_headway",
    "a_idm",
]


def _idm_accel(v, vl, gap, params, eps=1e-3):
    """Textbook Treiber IDM acceleration (same formula as fit_idm_per_driver)."""
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
    acc = a * (1.0 - free_term - (s_star / s) ** 2)
    return acc


def _load_driver_idm(idm_dir, driver_id):
    """Load per-driver IDM params from ``<idm_dir>/<T*>/idm.json``."""
    fp = os.path.join(idm_dir, driver_id, "idm.json")
    if not os.path.isfile(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("parameters")


def _discover_following_csvs(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if fn == "driving_data.csv" or re.match(r"segment_\d+\.csv$", fn):
                out.append(os.path.join(root, fn))
    return sorted(out)


def _extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    return m.group(1) if m else "UNKNOWN"


def _sorted_rows_times(rows, time_column, fallback_column):
    ts = [_parse_float(r.get(time_column)) for r in rows]
    if all(t is not None for t in ts):
        order = sorted(range(len(rows)), key=lambda i: ts[i])
        return [rows[i] for i in order], [ts[i] for i in order]
    ts_fb = [_parse_float(r.get(fallback_column)) for r in rows]
    if all(t is not None for t in ts_fb):
        order = sorted(range(len(rows)), key=lambda i: ts_fb[i])
        return [rows[i] for i in order], [ts_fb[i] for i in order]
    return None, None


def _features_at(rows, idx, a_idm_series):
    """Assemble one feature vector at row ``idx`` (requires a_idm_series[idx])."""
    r = rows[idx]
    v = _row_value(r, "ego_v_long")
    a = _row_value(r, "ego_a_long")
    g = _row_value(r, "distance_headway")
    rv = _row_value(r, "relative_v_long")
    lv = _row_value(r, "lead_v_long")
    if v is None or a is None or g is None or rv is None or lv is None:
        return None
    i_ttc = reciprocal_inv_feature(_parse_float(r.get("ttc")))
    i_thw = reciprocal_inv_feature(_parse_float(r.get("time_headway")))
    return [float(v), float(a), float(g), float(rv), float(lv),
            float(i_ttc), float(i_thw), float(a_idm_series[idx])]


def _build_segment(csv_path, idm_params, seq_len, time_column, time_fallback_column,
                   min_sim_time_s):
    """Return (X, y_residual, y_acceleration, v_current_series, dt_series)
    for causal windows in one CSV. All arrays aligned on prediction rows."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    rows = [hydrate_bc_gru_row_aliases(r) for r in rows]
    if not rows:
        return None

    rows, ts = _sorted_rows_times(rows, time_column, time_fallback_column)
    if rows is None:
        return None

    n = len(rows)
    if n <= seq_len + 1:
        return None

    a_idm_series = [0.0] * n
    for i in range(n):
        v = _row_value(rows[i], "ego_v_long") or 0.0
        vl = _row_value(rows[i], "lead_v_long") or 0.0
        g = _row_value(rows[i], "distance_headway")
        if g is None or g <= 0.0:
            a_idm_series[i] = 0.0
        else:
            a_idm_series[i] = _idm_accel(float(v), float(vl), float(g), idm_params)

    X_list, y_res, y_acc, v_cur, dt_list = [], [], [], [], []
    for t in range(seq_len, n):
        if min_sim_time_s is not None and min_sim_time_s > 0:
            if ts[t] is None or ts[t] < min_sim_time_s:
                continue

        window = []
        ok = True
        for k in range(t - seq_len, t):
            fv = _features_at(rows, k, a_idm_series)
            if fv is None:
                ok = False
                break
            window.append(fv)
        if not ok:
            continue

        a_real = _row_value(rows[t], "ego_a_long")
        v_now = _row_value(rows[t - 1], "ego_v_long")
        if a_real is None or v_now is None:
            continue

        residual = float(a_real) - float(a_idm_series[t])
        dt = 0.05
        if ts[t] is not None and ts[t - 1] is not None:
            dt = max(0.01, float(ts[t]) - float(ts[t - 1]))

        X_list.append(window)
        y_res.append(residual)
        y_acc.append(float(a_real))
        v_cur.append(float(v_now))
        dt_list.append(float(dt))

    if not X_list:
        return None
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_res, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_acc, dtype=np.float32).reshape(-1, 1),
        np.asarray(v_cur, dtype=np.float32).reshape(-1, 1),
        np.asarray(dt_list, dtype=np.float32).reshape(-1, 1),
    )


def _split_driver_paths(paths_by_driver, drivers, train_ratio, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for d in drivers:
        paths = list(paths_by_driver.get(d, []))
        rng.shuffle(paths)
        n = len(paths)
        if n == 0:
            continue
        n_tr = max(1, int(round(n * train_ratio)))
        n_va = max(1, int(round(n * val_ratio))) if n > 2 else 0
        n_te = max(0, n - n_tr - n_va)
        train += paths[:n_tr]
        val += paths[n_tr:n_tr + n_va]
        test += paths[n_tr + n_va:n_tr + n_va + n_te]
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--idm_dir", type=str, required=True,
                    help="Directory of per-driver IDM fits (<T*>/idm.json).")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_drivers", type=str, default="")
    ap.add_argument("--val_drivers", type=str, default="")
    ap.add_argument("--test_drivers", type=str, default="")
    ap.add_argument("--split_within_driver", action="store_true",
                    help="Split each driver's files into train/val/test by ratio.")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_v", type=float, default=0.1,
                    help="Speed consistency loss weight.")
    ap.add_argument("--lambda_jerk", type=float, default=0.05,
                    help="Residual jerk smoothness weight (0 disables).")
    ap.add_argument("--residual_clip", type=float, default=4.0,
                    help="Clip network output |Δa| to this during loss/metrics.")
    ap.add_argument("--min_sim_time_s", type=float, default=15.0,
                    help="Drop rows with sim_time_s < this (startup filter).")
    ap.add_argument("--time_column", type=str, default="sim_time_s")
    ap.add_argument("--time_fallback_column", type=str, default="timestamp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )

    # --- Discover CSVs, group by driver, match with IDM params ---
    all_paths = _discover_following_csvs(args.data_dir)
    paths_by_driver = {}
    for p in all_paths:
        d = _extract_driver_id(p)
        paths_by_driver.setdefault(d, []).append(p)

    def _parse_ids(s):
        return [x.strip() for x in str(s).split(",") if x.strip()]

    tr_drv = _parse_ids(args.train_drivers) or sorted(paths_by_driver.keys())
    va_drv = _parse_ids(args.val_drivers) or tr_drv
    te_drv = _parse_ids(args.test_drivers) or va_drv

    if args.split_within_driver:
        drv_union = sorted(set(tr_drv) | set(va_drv) | set(te_drv))
        train_paths, val_paths, test_paths = _split_driver_paths(
            paths_by_driver, drv_union,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
        )
    else:
        train_paths = [p for d in tr_drv for p in paths_by_driver.get(d, [])]
        val_paths = [p for d in va_drv for p in paths_by_driver.get(d, [])]
        test_paths = [p for d in te_drv for p in paths_by_driver.get(d, [])]

    # --- Build samples with per-driver IDM ---
    def _build_set(paths, label):
        Xs, ys_res, ys_acc, vs, dts = [], [], [], [], []
        for p in paths:
            d = _extract_driver_id(p)
            idm = _load_driver_idm(args.idm_dir, d)
            if idm is None:
                print("[WARN] no IDM for {}: {}".format(d, p))
                continue
            pack = _build_segment(p, idm, args.seq_len, args.time_column,
                                  args.time_fallback_column, args.min_sim_time_s)
            if pack is None:
                continue
            X, yr, ya, vc, dt = pack
            Xs.append(X)
            ys_res.append(yr)
            ys_acc.append(ya)
            vs.append(vc)
            dts.append(dt)
        if not Xs:
            return None
        X = np.concatenate(Xs, axis=0)
        yr = np.concatenate(ys_res, axis=0)
        ya = np.concatenate(ys_acc, axis=0)
        vc = np.concatenate(vs, axis=0)
        dt = np.concatenate(dts, axis=0)
        print("[{}] files={} samples={} X={} residual_std={:.4f} accel_std={:.4f}".format(
            label, len(Xs), X.shape[0], X.shape, float(yr.std()), float(ya.std())))
        return X, yr, ya, vc, dt

    train_set = _build_set(train_paths, "train")
    val_set = _build_set(val_paths, "val")
    test_set = _build_set(test_paths, "test") if test_paths else None
    if train_set is None or val_set is None:
        raise SystemExit("[ERR] empty train/val set.")

    X_tr, yr_tr, ya_tr, vc_tr, dt_tr = train_set

    # --- Feature normalization over the training set ---
    input_dim = X_tr.shape[-1]
    flat = X_tr.reshape(-1, input_dim)
    feat_mean = flat.mean(axis=0)
    feat_std = flat.std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    def _norm(X):
        return (X - feat_mean) / feat_std

    def _to_tensor(pack):
        X, yr, ya, vc, dt = pack
        return (
            torch.from_numpy(_norm(X)).float(),
            torch.from_numpy(yr).float(),
            torch.from_numpy(ya).float(),
            torch.from_numpy(vc).float(),
            torch.from_numpy(dt).float(),
        )

    tr_t = _to_tensor(train_set)
    va_t = _to_tensor(val_set)
    te_t = _to_tensor(test_set) if test_set else None

    tr_loader = DataLoader(TensorDataset(*tr_t), batch_size=args.batch_size,
                           shuffle=True, drop_last=False)
    va_loader = DataLoader(TensorDataset(*va_t), batch_size=args.batch_size,
                           shuffle=False, drop_last=False)
    te_loader = (DataLoader(TensorDataset(*te_t), batch_size=args.batch_size,
                            shuffle=False, drop_last=False) if te_t else None)

    # --- Model: GRU encoder -> linear head outputting Δa ---
    class ResidualGRU(nn.Module):
        def __init__(self, d_in, d_hid, n_layers, dropout):
            super(ResidualGRU, self).__init__()
            self.gru = nn.GRU(
                input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
                batch_first=True,
                dropout=(dropout if n_layers > 1 else 0.0),
            )
            self.head = nn.Sequential(
                nn.Linear(d_hid, d_hid // 2),
                nn.ReLU(),
                nn.Linear(d_hid // 2, 1),
            )

        def forward(self, x):
            out, _ = self.gru(x)
            h = out[:, -1, :]
            return self.head(h)

    model = ResidualGRU(input_dim, args.hidden_size, args.num_layers, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    resid_clip = float(args.residual_clip)

    def _run_epoch(loader, train):
        model.train(train)
        tot_loss = tot_res = tot_v = tot_j = 0.0
        n = 0
        prev_batch_pred = None
        for X, yr, ya, vc, dt in loader:
            X = X.to(device); yr = yr.to(device)
            ya = ya.to(device); vc = vc.to(device); dt = dt.to(device)

            pred = model(X)  # (B,1) residual
            pred_clip = torch.clamp(pred, -resid_clip, resid_clip)

            # Main residual loss
            loss_res = mse(pred, yr)

            # Speed consistency: v_next_real vs v_next_pred
            # pred acceleration = a_idm + delta_a (unclipped so grad flows)
            # a_idm is in the last feature before normalization; reconstruct from
            # the raw target: a_real - yr == a_idm at row t.
            a_idm_val = ya - yr  # (B,1)
            a_pred = a_idm_val + pred_clip
            v_next_pred = vc + a_pred * dt
            v_next_real = vc + ya * dt
            loss_v = mse(v_next_pred, v_next_real)

            # Jerk smoothness on residual (within batch neighboring preds are not
            # temporally adjacent, so this is a soft regularizer only)
            if args.lambda_jerk > 0:
                loss_j = torch.mean(pred_clip ** 2) * 0.0 + mse(pred_clip, torch.zeros_like(pred_clip)) * 0.0
                # Proper jerk would need temporal adjacency; use simple L2 on
                # residual magnitude to discourage large corrections.
                loss_j = torch.mean(pred_clip ** 2)
            else:
                loss_j = torch.zeros((), device=device)

            loss = loss_res + args.lambda_v * loss_v + args.lambda_jerk * loss_j

            if train:
                opt.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            bs = X.size(0)
            tot_loss += loss.item() * bs
            tot_res += loss_res.item() * bs
            tot_v += loss_v.item() * bs
            tot_j += loss_j.item() * bs
            n += bs
        return tot_loss / max(1, n), tot_res / max(1, n), tot_v / max(1, n), tot_j / max(1, n)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history = []
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_r, tr_v, tr_j = _run_epoch(tr_loader, True)
        va_loss, va_r, va_v, va_j = _run_epoch(va_loader, False)
        history.append(dict(epoch=ep, train=tr_loss, val=va_loss,
                            train_res=tr_r, val_res=va_r,
                            train_v=tr_v, val_v=va_v,
                            train_j=tr_j, val_j=va_j))
        print("ep {:3d}: train={:.5f} (res={:.4f} v={:.4f} j={:.4f}) "
              "val={:.5f} (res={:.4f} v={:.4f} j={:.4f})".format(
                  ep, tr_loss, tr_r, tr_v, tr_j, va_loss, va_r, va_v, va_j))

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = ep
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping at epoch {}.".format(ep))
                break

    # --- Test metrics ---
    if te_loader:
        model.load_state_dict(torch.load(best_path, map_location=device))
        te_loss, te_r, te_v, te_j = _run_epoch(te_loader, False)
        print("TEST loss={:.5f} res_MSE={:.5f} v_MSE={:.5f}".format(te_loss, te_r, te_v))
        test_report = dict(loss=te_loss, residual_mse=te_r, v_consistency_mse=te_v)
    else:
        test_report = None

    # --- Save meta / normalization ---
    meta = dict(
        arch="residual_gru",
        features=RESIDUAL_FEATURES,
        targets=["ego_a_long"],
        target_mode="residual_delta_a",
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        input_dim=input_dim,
        idm_dir=os.path.abspath(args.idm_dir),
        min_sim_time_s=args.min_sim_time_s,
        residual_clip=resid_clip,
        lambda_v=args.lambda_v,
        lambda_jerk=args.lambda_jerk,
        train_drivers=tr_drv,
        val_drivers=va_drv,
        test_drivers=te_drv,
        split_within_driver=bool(args.split_within_driver),
    )
    with open(os.path.join(args.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    report = dict(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        test=test_report,
        feature_mean=feat_mean.tolist(),
        feature_std=feat_std.tolist(),
        history=history,
    )
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[OK] saved best={} meta/report in {}".format(best_path, args.out_dir))


if __name__ == "__main__":
    main()
