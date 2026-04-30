# -*- coding: utf-8 -*-
"""
Train a GRU behavior-cloning policy for car-following imitation learning.

Input data layout:
  <data_dir>/T*/行车/<session>/segment_XXX.csv

Default target:
  predict longitudinal acceleration at t:
    [ego_a_long]

for i in $(seq 1 20); do
  D="T${i}"
  python /home/zwx/driver_model/train/train_bc_gru.py \
    --data_dir /home/zwx/driver_model/outputs/following_il_clean_gap04 \
    --out_dir /home/zwx/driver_model/outputs/il_bc_gru_per_driver/${D}_longitudinal_framewin \
    --train_drivers ${D} \
    --val_drivers ${D} \
    --test_drivers ${D} \
    --split_within_driver \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seq_len 20 \
    --target_weights 1.0 \
    --epochs 60
done
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import random
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_FEATURES = [
    "dt_prev",
    "ego_v_long",
    "ego_a_long",
    "distance_headway",
    "relative_v_long",
    "lead_v_long",
    "ttc",
    "ttc_valid",
    "time_headway",
    "time_headway_valid",
]

DEFAULT_TARGETS = ["ego_a_long"]


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


def _discover_segment_csvs(data_dir):
    out = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if not fn.endswith(".csv"):
                continue
            if not re.match(r"segment_\d+\.csv$", fn):
                continue
            out.append(os.path.join(root, fn))
    return sorted(out)


def _extract_driver_id(path):
    p = path.replace("\\", "/")
    m = re.search(r"/(T\d+)(?:/|$)", p)
    if m:
        return m.group(1)
    return "UNKNOWN"


def _row_value(row, key):
    if key in row:
        return _parse_float(row.get(key))
    if key == "relative_speed":
        lv = _parse_float(row.get("lead_speed"))
        ev = _parse_float(row.get("ego_speed"))
        if lv is None or ev is None:
            return None
        return lv - ev
    return None


def _build_segment_arrays(csv_path, features, targets):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return None, None

    ts = [_parse_float(r.get("timestamp")) for r in rows]

    x = []
    y = []
    for i, r in enumerate(rows):
        fv = []
        ok = True
        for k in features:
            if k == "dt_prev":
                if i == 0:
                    v = 0.0
                else:
                    t0 = ts[i - 1]
                    t1 = ts[i]
                    if t0 is None or t1 is None:
                        v = None
                    else:
                        # Explicitly provide per-frame elapsed time to handle non-uniform sampling.
                        v = max(0.0, t1 - t0)
            else:
                v = _row_value(r, k)
            if v is None:
                ok = False
                break
            fv.append(v)
        if not ok:
            continue
        tv = []
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


def _build_samples(segment_paths, features, targets, seq_len):
    xs = []
    ys = []
    meta = []
    for p in segment_paths:
        arr_x, arr_y = _build_segment_arrays(p, features, targets)
        if arr_x is None:
            continue
        n = arr_x.shape[0]
        if n < seq_len:
            continue
        for t in range(seq_len - 1, n):
            xs.append(arr_x[t - seq_len + 1 : t + 1])
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


class BCGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(BCGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.head(h)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/following_il_clean_gap04",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/home/zwx/driver_model/outputs/il_bc_gru",
    )
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
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
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    segment_paths = _discover_segment_csvs(args.data_dir)
    if args.max_segments and args.max_segments > 0:
        segment_paths = segment_paths[: args.max_segments]
    if not segment_paths:
        raise RuntimeError("No segment_*.csv found under {}".format(args.data_dir))

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

    features = list(DEFAULT_FEATURES)
    targets = list(DEFAULT_TARGETS)
    target_weights = [float(x.strip()) for x in args.target_weights.split(",") if x.strip()]
    if len(target_weights) != len(targets):
        raise RuntimeError(
            "target_weights length {} must match targets length {}".format(
                len(target_weights), len(targets)
            )
        )
    train_x, train_y, _ = _build_samples(train_paths, features, targets, args.seq_len)
    val_x, val_y, _ = _build_samples(val_paths, features, targets, args.seq_len)
    test_x, test_y, _ = _build_samples(test_paths, features, targets, args.seq_len)
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

    model = BCGRU(
        input_dim=len(features),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
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
        "data_dir": args.data_dir,
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
                "features": features,
                "targets": targets,
                "seq_len": args.seq_len,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "target_weights": target_weights,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("[OK] model:", best_path)
    print("[OK] report:", os.path.join(args.out_dir, "train_report.json"))


if __name__ == "__main__":
    main()

