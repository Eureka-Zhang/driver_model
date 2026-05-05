# -*- coding: utf-8 -*-
"""
Train a **TCN** (Temporal Convolutional Network) behavior-cloning policy for car-following IL.

Same data pipeline, features, and outputs as ``train_bc_gru.py``; only the sequence encoder
changes from GRU to dilated causal convolutions (Bai et al.–style temporal blocks).

Default input per timestep (``bc_gru_features.DEFAULT_FEATURES``, no ``dt_prev`` unless
``--with-dt-prev``).

Example::

  python3 following/gru_train/train_bc_tcn.py \\
    --data_dir following/outputs/following_calibrated \\
    --out_dir following/outputs/il_bc_tcn_per_driver/T5_longitudinal_framewin \\
    --train_drivers T5 --val_drivers T5 --test_drivers T5 \\
    --split_within_driver \\
    --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 \\
    --seq_len 20 --epochs 60 \\
    --tcn_channels 64,64,64 --tcn_kernel_size 3

Batch (one run per driver, default ``--split_within_driver``)::

  python3 following/gru_train/train_bc_tcn.py \\
    --data_dir following/outputs/following_calibrated \\
    --out_dir following/outputs/il_bc_tcn_per_driver/{driver} \\
    --batch_drivers ALL \\
    --seq_len 20 --epochs 60 \\
    --tcn_channels 64,64,64 --tcn_kernel_size 3

Use ``--batch_drivers T1,T3,T5`` for a subset; if ``--out_dir`` has no ``{driver}``,
a subdirectory named after each driver is appended.

**Rollout** (closed loop on calibrated / segment CSVs): use
``generate_no_driver_following_outputs_tcn.py`` with the same ``model_dir`` as training
(``best_model.pt`` + ``model_meta.json`` + ``train_report.json``).
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
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from bc_gru_features import (
    DEFAULT_FEATURES,
    DEFAULT_FEATURES_LEGACY,
    DEFAULT_TARGETS,
    _parse_float,
    features_at_timestep,
    hydrate_bc_gru_row_aliases,
)


def _discover_training_csvs(data_dir):
    """``driving_data.csv`` (calibrated sessions) or ``segment_<n>.csv`` (clean IL segments)."""
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
    if m:
        return m.group(1)
    return "UNKNOWN"


def _sorted_rows_times(rows, time_column, fallback_column):
    ts = [_parse_float(r.get(time_column)) for r in rows]
    if all(t is not None for t in ts):
        order = sorted(range(len(rows)), key=lambda i: ts[i])
        rows_s = [rows[i] for i in order]
        ts_s = [ts[i] for i in order]
        return rows_s, ts_s
    if fallback_column != time_column:
        ts_fb = [_parse_float(r.get(fallback_column)) for r in rows]
        if all(t is not None for t in ts_fb):
            print(
                "[WARN] column {!r} missing on some rows — sorting with fallback {!r}".format(
                    time_column,
                    fallback_column,
                )
            )
            order = sorted(range(len(rows)), key=lambda i: ts_fb[i])
            rows_s = [rows[i] for i in order]
            ts_s = [ts_fb[i] for i in order]
            return rows_s, ts_s
    return None, None


def _build_segment_arrays(csv_path, features, targets, time_column, time_fallback_column):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(dict(r))
    rows = [hydrate_bc_gru_row_aliases(r) for r in rows]
    if not rows:
        return None, None

    rows, ts = _sorted_rows_times(rows, time_column, time_fallback_column)
    if rows is None:
        print("[WARN] skip (no monotone time column): {}".format(csv_path))
        return None, None

    x = []
    y = []
    for i, r in enumerate(rows):
        fv = features_at_timestep(rows, ts, i, features)
        if fv is None:
            continue
        tv = []
        ok = True
        for k in targets:
            v = _parse_float(r.get(k))
            if v is None:
                ok = False
                break
            tv.append(v)
        if not ok:
            continue
        x.append(fv)
        y.append(tv)

    if not x:
        return None, None
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _build_samples(segment_paths, features, targets, seq_len, time_column, time_fallback_column):
    xs = []
    ys = []
    meta = []
    for p in segment_paths:
        arr_x, arr_y = _build_segment_arrays(
            p, features, targets, time_column, time_fallback_column
        )
        if arr_x is None:
            continue
        n = arr_x.shape[0]
        if n <= seq_len:
            continue
        for t in range(seq_len, n):
            xs.append(arr_x[t - seq_len : t])
            ys.append(arr_y[t])
            meta.append(p)
    if not xs:
        return None, None, []
    return np.stack(xs), np.stack(ys), meta


def _split_drivers(all_drivers, seed):
    drivers = sorted(list(all_drivers))
    rng = random.Random(seed)
    rng.shuffle(drivers)
    n = len(drivers)
    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = drivers[:n_train]
    val = drivers[n_train : n_train + n_val]
    test = drivers[n_train + n_val :]
    if not test:
        test = [drivers[-1]]
        if drivers[-1] in val:
            val = val[:-1]
    return train, val, test


def _split_paths_within_each_driver(by_driver, selected_drivers, seed, train_ratio, val_ratio):
    train_paths, val_paths, test_paths = [], [], []
    rng = random.Random(seed)
    for d in selected_drivers:
        paths = list(by_driver.get(d, []))
        if not paths:
            continue
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 2:
            if n_val > 0:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            n_test = n - n_train - n_val

        train_paths.extend(paths[:n_train])
        val_paths.extend(paths[n_train : n_train + n_val])
        test_paths.extend(paths[n_train + n_val :])
    return train_paths, val_paths, test_paths


class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CausalConv1d(nn.Module):
    """Left-padded 1-D conv; output length matches input length (causal receptive field)."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x):
        # x: [B, C, T]
        if self._pad > 0:
            x = F.pad(x, (self._pad, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """
    Two causal conv layers with same dilation, residual add, ReLU + dropout.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BCTCN(nn.Module):
    """
    Dilated TCN over the time axis; last frame is mapped to acceleration like the GRU model.
    Input: ``[B, T, input_dim]``.
    """

    def __init__(self, input_dim, channel_dims, kernel_size, dropout, output_dim):
        super(BCTCN, self).__init__()
        if not channel_dims:
            raise ValueError("channel_dims must be non-empty")
        layers = []
        for i, out_c in enumerate(channel_dims):
            dilation = 2 ** i
            in_c = input_dim if i == 0 else channel_dims[i - 1]
            layers.append(TemporalBlock(in_c, out_c, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channel_dims[-1], output_dim)

    def forward(self, x):
        # [B, T, F] -> [B, F, T]
        z = x.transpose(1, 2)
        z = self.tcn(z)
        last = z[:, :, -1]
        return self.head(last)


def _weighted_mse(pred, target, weights):
    diff2 = (pred - target) ** 2
    return torch.mean(diff2 * weights)


def _run_epoch(model, loader, optimizer, device, action_weights):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if training:
            optimizer.zero_grad()
        pred = model(xb)
        loss = _weighted_mse(pred, yb, action_weights)
        if training:
            loss.backward()
            optimizer.step()
        b = xb.shape[0]
        total_loss += loss.item() * b
        total_n += b
    return total_loss / max(1, total_n)


def _eval_metrics(model, loader, device):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            gts.append(yb.numpy())
    if not preds:
        return {}
    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    mse = np.mean((pred - gt) ** 2, axis=0)
    mae = np.mean(np.abs(pred - gt), axis=0)
    return {"mse": mse.tolist(), "mae": mae.tolist(), "n": int(pred.shape[0])}


def _parse_tcn_channels(s):
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise ValueError("tcn_channels must list at least one integer, e.g. 64,64,64")
    return [int(p) for p in parts]


def _driver_sort_key(d):
    if d.startswith("T") and d[1:].isdigit():
        return int(d[1:])
    return 99999


def run_bc_tcn_training(args):
    """One full train/val/test job; ``args`` is an argparse.Namespace with all CLI fields."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    channel_dims = _parse_tcn_channels(args.tcn_channels)

    os.makedirs(args.out_dir, exist_ok=True)
    segment_paths = _discover_training_csvs(args.data_dir)
    if args.max_segments and args.max_segments > 0:
        segment_paths = segment_paths[: args.max_segments]
    if not segment_paths:
        raise RuntimeError(
            "No driving_data.csv or segment_*.csv found under {}".format(args.data_dir)
        )

    by_driver = {}
    for p in segment_paths:
        d = _extract_driver_id(p)
        by_driver.setdefault(d, []).append(p)

    all_drivers = set(by_driver.keys())
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.test_ratio <= 0:
        raise RuntimeError("Ratios must satisfy train>0, val>=0, test>0.")
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise RuntimeError("train_ratio + val_ratio + test_ratio must equal 1.0")

    def _parse_ids(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    if args.train_drivers or args.val_drivers or args.test_drivers:
        train_drivers = _parse_ids(args.train_drivers)
        val_drivers = _parse_ids(args.val_drivers)
        test_drivers = _parse_ids(args.test_drivers)
    else:
        train_drivers, val_drivers, test_drivers = _split_drivers(all_drivers, args.seed)

    if args.split_within_driver:
        selected_drivers = sorted(
            set(train_drivers) | set(val_drivers) | set(test_drivers)
        )
        if not selected_drivers:
            selected_drivers = sorted(all_drivers)
        train_paths, val_paths, test_paths = _split_paths_within_each_driver(
            by_driver=by_driver,
            selected_drivers=selected_drivers,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        train_drivers = selected_drivers
        val_drivers = selected_drivers
        test_drivers = selected_drivers
    else:
        train_paths = [p for d in train_drivers for p in by_driver.get(d, [])]
        val_paths = [p for d in val_drivers for p in by_driver.get(d, [])]
        test_paths = [p for d in test_drivers for p in by_driver.get(d, [])]

    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError(
            "Split invalid. train/val/test sizes: {}/{}/{}".format(
                len(train_paths), len(val_paths), len(test_paths)
            )
        )

    features = list(DEFAULT_FEATURES_LEGACY if args.with_dt_prev else DEFAULT_FEATURES)
    targets = list(DEFAULT_TARGETS)
    target_weights = [float(x.strip()) for x in args.target_weights.split(",") if x.strip()]
    if len(target_weights) != len(targets):
        raise RuntimeError(
            "target_weights length {} must match targets length {}".format(
                len(target_weights), len(targets)
            )
        )
    train_x, train_y, _ = _build_samples(
        train_paths,
        features,
        targets,
        args.seq_len,
        args.time_column,
        args.time_fallback_column,
    )
    val_x, val_y, _ = _build_samples(
        val_paths,
        features,
        targets,
        args.seq_len,
        args.time_column,
        args.time_fallback_column,
    )
    test_x, test_y, _ = _build_samples(
        test_paths,
        features,
        targets,
        args.seq_len,
        args.time_column,
        args.time_fallback_column,
    )
    if train_x is None or val_x is None or test_x is None:
        raise RuntimeError("No valid samples after windowing. Check seq_len / data quality.")

    feat_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    feat_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)
    train_x = (train_x - feat_mean) / feat_std
    val_x = (val_x - feat_mean) / feat_std
    test_x = (test_x - feat_mean) / feat_std

    train_ds = SeqDataset(train_x.astype(np.float32), train_y.astype(np.float32))
    val_ds = SeqDataset(val_x.astype(np.float32), val_y.astype(np.float32))
    test_ds = SeqDataset(test_x.astype(np.float32), test_y.astype(np.float32))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = BCTCN(
        input_dim=len(features),
        channel_dims=channel_dims,
        kernel_size=args.tcn_kernel_size,
        dropout=args.dropout,
        output_dim=len(targets),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    action_weights = torch.tensor(target_weights, dtype=torch.float32, device=device)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "best_model.pt")

    history = []
    for ep in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, device, action_weights)
        val_loss = _run_epoch(model, val_loader, None, device, action_weights)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        print(
            "epoch {:03d} train_loss {:.6f} val_loss {:.6f}".format(
                ep, train_loss, val_loss
            )
        )
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping at epoch {}.".format(ep))
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = _eval_metrics(model, test_loader, device)
    print("best epoch:", best_epoch, "best val:", best_val)
    print("test metrics:", test_metrics)

    cfg = {
        "arch": "tcn",
        "data_dir": args.data_dir,
        "time_column": args.time_column,
        "time_fallback_column": args.time_fallback_column,
        "with_dt_prev": bool(args.with_dt_prev),
        "seq_len": args.seq_len,
        "features": features,
        "targets": targets,
        "train_drivers": train_drivers,
        "val_drivers": val_drivers,
        "test_drivers": test_drivers,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "tcn_channels": channel_dims,
        "tcn_kernel_size": args.tcn_kernel_size,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "history": history,
        "split_within_driver": bool(args.split_within_driver),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_train_segments": len(train_paths),
        "n_val_segments": len(val_paths),
        "n_test_segments": len(test_paths),
        "target_weights": target_weights,
    }
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "arch": "tcn",
                "features": features,
                "targets": targets,
                "seq_len": args.seq_len,
                "tcn_channels": channel_dims,
                "tcn_kernel_size": args.tcn_kernel_size,
                "dropout": args.dropout,
                "target_weights": target_weights,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[OK] model:", best_path)
    print("[OK] report:", os.path.join(args.out_dir, "train_report.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/following_calibrated",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/following/outputs/il_bc_tcn",
    )
    ap.add_argument("--time_column", type=str, default="sim_time_s")
    ap.add_argument(
        "--time_fallback_column",
        type=str,
        default="timestamp",
        help="Fallback time column when --time_column values are missing.",
    )
    ap.add_argument(
        "--with-dt-prev",
        action="store_true",
        help="Legacy 8-D input: include dt_prev as first feature.",
    )
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument(
        "--tcn_channels",
        type=str,
        default="64,64,64",
        help="Comma-separated channel width per temporal block (e.g. 64,64,64)",
    )
    ap.add_argument("--tcn_kernel_size", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--target_weights",
        type=str,
        default="1.0",
        help="Comma-separated weights aligned with targets",
    )
    ap.add_argument("--train_drivers", type=str, default="")
    ap.add_argument("--val_drivers", type=str, default="")
    ap.add_argument("--test_drivers", type=str, default="")
    ap.add_argument(
        "--split_within_driver",
        action="store_true",
        help="Randomly split segments into train/val/test within each selected driver",
    )
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--max_segments", type=int, default=0)
    ap.add_argument(
        "--batch_drivers",
        type=str,
        default="",
        help="Comma-separated driver IDs (e.g. T1,T5). Use ALL for every driver that has CSVs. "
        "Runs one independent job per driver with train/val/test set to that driver; "
        "forces --split_within_driver unless --batch_no_split_within.",
    )
    ap.add_argument(
        "--batch_no_split_within",
        action="store_true",
        help="In batch mode, do not force --split_within_driver (use your explicit flags).",
    )
    args = ap.parse_args()

    batch_spec = str(args.batch_drivers).strip()
    if batch_spec:
        segment_paths = _discover_training_csvs(args.data_dir)
        if args.max_segments and args.max_segments > 0:
            segment_paths = segment_paths[: args.max_segments]
        if not segment_paths:
            print(
                "No driving_data.csv or segment_*.csv found under {}".format(args.data_dir),
                file=sys.stderr,
            )
            return 2
        by_driver = {}
        for p in segment_paths:
            d = _extract_driver_id(p)
            by_driver.setdefault(d, []).append(p)
        if batch_spec.upper() == "ALL":
            driver_ids = sorted(by_driver.keys(), key=_driver_sort_key)
        else:
            driver_ids = [x.strip() for x in batch_spec.split(",") if x.strip()]
        if not driver_ids:
            print("[ERROR] --batch_drivers produced an empty list.", file=sys.stderr)
            return 2
        base_out = args.out_dir
        n_ok = 0
        n_fail = 0
        for drv in driver_ids:
            if drv not in by_driver:
                print("[WARN] skip {} (no CSV under --data_dir)".format(drv))
                n_fail += 1
                continue
            sub = argparse.Namespace(**vars(args))
            sub.train_drivers = drv
            sub.val_drivers = drv
            sub.test_drivers = drv
            if not args.batch_no_split_within:
                sub.split_within_driver = True
            if "{driver}" in sub.out_dir:
                sub.out_dir = sub.out_dir.replace("{driver}", drv)
            else:
                sub.out_dir = os.path.join(base_out, drv)
            print("\n===== batch driver {} out_dir {} =====".format(drv, sub.out_dir))
            try:
                run_bc_tcn_training(sub)
                n_ok += 1
            except RuntimeError as err:
                print("[FAIL] {} {}".format(drv, err), file=sys.stderr)
                n_fail += 1
            except Exception as err:
                print("[FAIL] {} {}".format(drv, err), file=sys.stderr)
                n_fail += 1
        print(
            "\nBatch done: {} succeeded, {} failed/skipped.".format(n_ok, n_fail),
            file=sys.stderr if n_fail else sys.stdout,
        )
        return 0 if n_fail == 0 else 1

    try:
        run_bc_tcn_training(args)
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
