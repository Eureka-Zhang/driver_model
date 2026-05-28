#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select typical drivers per style using D_rep, S_stability, P_salience, and T_typical.

Outputs:
  <out_dir>/following_style_typical_drivers.csv   (task=following)
  <out_dir>/overtaking_style_typical_drivers.csv  (task=overtaking)
  <out_dir>/*_typical_selection.json

Usage::

  python3 following/scripts/select_typical_drivers_by_style.py --task following
  python3 following/scripts/select_typical_drivers_by_style.py --task overtaking
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_FOLLOWING_CLUSTER = _ROOT / "following" / "scripts" / "cluster_following_style.py"
_OVERTAKING_CLUSTER = _ROOT / "overtaking" / "scripts" / "cluster_overtaking_style.py"

_STYLE_ORDER = ("conservative", "neutral", "aggressive")


def _load_py_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load {}".format(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm_driver(s: str) -> str:
    s = str(s).strip().upper()
    m = re.match(r"^T(\d+)$", s)
    return "T{}".format(int(m.group(1))) if m else s


def _weighted_dist2(z: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> float:
    d = z - ref
    return float(np.sum(weights * d * d))


def _zscore_global(rows: Sequence[Dict[str, float]], keys: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    m = len(keys)
    mat = np.zeros((len(rows), m), dtype=np.float64)
    for i, rec in enumerate(rows):
        for j, k in enumerate(keys):
            mat[i, j] = float(rec.get(k, 0.0))
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std[std < 1e-9] = 1.0
    return mean, std


def _zvec(raw: Dict[str, float], keys: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    m = len(keys)
    out = np.zeros(m, dtype=np.float64)
    for j, k in enumerate(keys):
        out[j] = (float(raw.get(k, 0.0)) - mean[j]) / std[j]
    return out


def _segment_stability(seg_z: List[np.ndarray], weights: np.ndarray) -> float:
    """S_i: mean segment deviation from the driver's own segment-mean z vector."""
    if not seg_z:
        return 0.0
    z_bar_i = np.mean(np.stack(seg_z, axis=0), axis=0)
    acc = 0.0
    for z_k in seg_z:
        acc += _weighted_dist2(z_k, z_bar_i, weights)
    return acc / len(seg_z)


def _read_labels_csv(path: Path, task: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if task == "following":
                drv = _norm_driver(row.get("driver") or row.get("driver_id") or "")
                sty = str(row.get("label") or row.get("style_label") or "").strip().lower()
            else:
                drv = _norm_driver(row.get("driver_id") or row.get("driver") or "")
                sty = str(row.get("style_label") or row.get("label") or "").strip().lower()
            if drv and sty in _STYLE_ORDER:
                labels[drv] = sty
    return labels


def _read_following_scores(path: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            drv = _norm_driver(row.get("driver") or "")
            if drv and row.get("score") not in (None, ""):
                scores[drv] = float(row["score"])
    return scores


def run_following(
    data_dir: Path,
    labels_csv: Path,
    out_dir: Path,
    weights: np.ndarray,
    lambdas: Tuple[float, float, float],
) -> None:
    cfs = _load_py_module("cluster_following_style", _FOLLOWING_CLUSTER)
    keys = list(cfs.ALL_SCORE_FEATURES)

    all_csv = cfs._discover_csvs(str(data_dir))
    by_driver: Dict[str, List[str]] = defaultdict(list)
    for fp in all_csv:
        by_driver[cfs._driver_id(fp)].append(fp)

    driver_raw: Dict[str, Dict[str, float]] = {}
    seg_rows: List[Tuple[str, Dict[str, float]]] = []
    for drv, paths in sorted(by_driver.items()):
        feat_drv = cfs._extract_driver_features(paths, min_speed=2.0, min_gap=1.0)
        if feat_drv is not None:
            driver_raw[drv] = feat_drv
        for fp in paths:
            feat_seg = cfs._extract_driver_features([fp], min_speed=2.0, min_gap=1.0)
            if feat_seg is not None:
                seg_rows.append((drv, feat_seg))

    if not driver_raw or not seg_rows:
        raise SystemExit("[ERR] no following driver/segment features")

    # z-score mean/std from driver-level aggregates (applied to z_i and z_{i,k})
    g_mean_drv, g_std_drv = _zscore_global(list(driver_raw.values()), keys)
    w = np.asarray(weights, dtype=np.float64)

    z_driver: Dict[str, np.ndarray] = {
        drv: _zvec(raw, keys, g_mean_drv, g_std_drv) for drv, raw in driver_raw.items()
    }
    z_by_driver_seg: Dict[str, List[np.ndarray]] = defaultdict(list)
    for drv, raw in seg_rows:
        z_by_driver_seg[drv].append(_zvec(raw, keys, g_mean_drv, g_std_drv))

    labels = _read_labels_csv(labels_csv, "following")
    scores = _read_following_scores(labels_csv)
    drivers = sorted(set(z_driver) & set(labels), key=lambda d: int(d[1:]))
    mean_score = float(np.mean([scores[d] for d in drivers if d in scores]))

    by_style: Dict[str, List[str]] = defaultdict(list)
    for d in drivers:
        by_style[labels[d]].append(d)

    rows_out = []
    summary = {
        "task": "following",
        "data_dir": str(data_dir.resolve()),
        "labels_csv": str(labels_csv.resolve()),
        "feature_keys": keys,
        "weights": weights.tolist(),
        "lambda": list(lambdas),
        "mean_score_f": mean_score,
        "by_style": {},
    }

    for sty in _STYLE_ORDER:
        members = [d for d in by_style.get(sty, []) if d in z_driver]
        if not members:
            continue
        z_bar_g = np.mean(np.stack([z_driver[d] for d in members], axis=0), axis=0)
        ranking = []
        for d in members:
            d_rep = _weighted_dist2(z_driver[d], z_bar_g, w)
            s_stab = _segment_stability(z_by_driver_seg[d], w)
            sc = float(scores.get(d, 0.0))
            p_sal = abs(sc - mean_score)
            if sty == "neutral":
                t_typ = lambdas[0] * d_rep + lambdas[1] * s_stab + lambdas[2] * p_sal
            else:
                t_typ = lambdas[0] * d_rep + lambdas[1] * s_stab - lambdas[2] * p_sal
            ranking.append(
                dict(
                    driver_id=d,
                    T_typical=t_typ,
                    D_rep=d_rep,
                    S_stability=s_stab,
                    P_salience=p_sal,
                    score=sc,
                    style_label=sty,
                )
            )
        ranking.sort(key=lambda x: x["T_typical"])
        for rnk, ent in enumerate(ranking, start=1):
            ent["rank"] = rnk
            rows_out.append(ent)
        summary["by_style"][sty] = {
            "style_label": sty,
            "prototype_driver": ranking[0]["driver_id"],
            "typicality_ranking": ranking,
            "drivers_in_style": members,
            "n_drivers": len(members),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "following_style_typical_drivers.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style_label",
                "rank",
                "driver_id",
                "T_typical",
                "D_rep",
                "S_stability",
                "P_salience",
                "score",
            ],
        )
        writer.writeheader()
        for ent in rows_out:
            writer.writerow(ent)

    json_path = out_dir / "following_style_typical_selection.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved] {}".format(csv_path))
    print("[saved] {}".format(json_path))
    for sty in _STYLE_ORDER:
        if sty in summary["by_style"]:
            b = summary["by_style"][sty]
            print(
                "  {}: prototype={}".format(sty, b["prototype_driver"]),
            )


def run_overtaking(
    data_dir: Path,
    labels_csv: Path,
    out_dir: Path,
    weights: np.ndarray,
    lambdas: Tuple[float, float, float],
    seg_kw: dict,
) -> None:
    ocs = _load_py_module("cluster_overtaking_style", _OVERTAKING_CLUSTER)
    keys = list(ocs.CLUSTER_FEATURES)

    maneuvers = ocs._discover_maneuvers(str(data_dir))
    by_driver: Dict[str, list] = defaultdict(list)
    for m in maneuvers:
        by_driver[ocs._driver_id(m["canonical_path"])].append(m)

    driver_raw: Dict[str, Dict[str, float]] = {}
    seg_rows: List[Tuple[str, Dict[str, float]]] = []
    for drv, mans in sorted(by_driver.items()):
        feats_acc = {k: [] for k in keys}
        for man in mans:
            rec, _n = ocs._features_maneuver(man, **seg_kw)
            if rec is None:
                continue
            raw = {k: float(rec[k]) for k in keys}
            seg_rows.append((drv, raw))
            for k in keys:
                feats_acc[k].append(raw[k])
        if feats_acc[keys[0]]:
            driver_raw[drv] = {k: float(np.median(feats_acc[k])) for k in keys}

    if not driver_raw or not seg_rows:
        raise SystemExit("[ERR] no overtaking driver/segment features")

    g_mean_drv, g_std_drv = _zscore_global(list(driver_raw.values()), keys)
    w = np.asarray(weights, dtype=np.float64)

    z_driver: Dict[str, np.ndarray] = {
        drv: _zvec(raw, keys, g_mean_drv, g_std_drv) for drv, raw in driver_raw.items()
    }
    z_by_driver_seg: Dict[str, List[np.ndarray]] = defaultdict(list)
    for drv, raw in seg_rows:
        z_by_driver_seg[drv].append(_zvec(raw, keys, g_mean_drv, g_std_drv))

    labels = _read_labels_csv(labels_csv, "overtaking")
    drivers = sorted(set(z_driver) & set(labels), key=lambda d: int(d[1:]))

    # Score^o on driver-level z (same linear combo as clustering narrative)
    score_coef = np.array([-1, -1, 1, 1, 1, -1], dtype=np.float64)
    scores: Dict[str, float] = {}
    for drv in drivers:
        scores[drv] = float(np.dot(score_coef, z_driver[drv]))
    mean_score = float(np.mean(list(scores.values())))

    by_style: Dict[str, List[str]] = defaultdict(list)
    for d in drivers:
        by_style[labels[d]].append(d)

    rows_out = []
    summary = {
        "task": "overtaking",
        "data_dir": str(data_dir.resolve()),
        "labels_csv": str(labels_csv.resolve()),
        "feature_keys": keys,
        "weights": weights.tolist(),
        "lambda": list(lambdas),
        "mean_score_o": mean_score,
        "by_style": {},
    }

    for sty in _STYLE_ORDER:
        members = [d for d in by_style.get(sty, []) if d in z_driver]
        if not members:
            continue
        z_bar_g = np.mean(np.stack([z_driver[d] for d in members], axis=0), axis=0)
        ranking = []
        for d in members:
            d_rep = _weighted_dist2(z_driver[d], z_bar_g, w)
            s_stab = _segment_stability(z_by_driver_seg[d], w)
            sc = scores[d]
            p_sal = abs(sc - mean_score)
            if sty == "neutral":
                t_typ = lambdas[0] * d_rep + lambdas[1] * s_stab + lambdas[2] * p_sal
            else:
                t_typ = lambdas[0] * d_rep + lambdas[1] * s_stab - lambdas[2] * p_sal
            ranking.append(
                dict(
                    driver_id=d,
                    T_typical=t_typ,
                    D_rep=d_rep,
                    S_stability=s_stab,
                    P_salience=p_sal,
                    score=sc,
                    style_label=sty,
                )
            )
        ranking.sort(key=lambda x: x["T_typical"])
        for rnk, ent in enumerate(ranking, start=1):
            ent["rank"] = rnk
            rows_out.append(ent)
        summary["by_style"][sty] = {
            "style_label": sty,
            "prototype_driver": ranking[0]["driver_id"],
            "typicality_ranking": ranking,
            "drivers_in_style": members,
            "n_drivers": len(members),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "overtaking_style_typical_drivers.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "style_label",
                "rank",
                "driver_id",
                "T_typical",
                "D_rep",
                "S_stability",
                "P_salience",
                "score",
            ],
        )
        writer.writeheader()
        for ent in rows_out:
            writer.writerow(ent)

    json_path = out_dir / "overtaking_style_typical_selection.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[saved] {}".format(csv_path))
    print("[saved] {}".format(json_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=("following", "overtaking"), required=True)
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--labels_csv", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--feature_weights", type=str, default="")
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    ap.add_argument("--lambda3", type=float, default=1.0)
    ap.add_argument("--right_y_min", type=float, default=-9.30)
    ap.add_argument("--right_y_max", type=float, default=-5.95)
    ap.add_argument("--ego_half_width_m", type=float, default=0.9)
    ap.add_argument("--segment_smooth_window", type=int, default=5)
    ap.add_argument("--edge_eps_m", type=float, default=0.15)
    ap.add_argument("--center_hold_sec", type=float, default=0.5)
    ap.add_argument("--y_center_p2_end", type=float, default=None)
    args = ap.parse_args()

    lambdas = (args.lambda1, args.lambda2, args.lambda3)

    if args.task == "following":
        data_dir = Path(args.data_dir or _ROOT / "following/outputs/following_calibrated")
        labels_csv = Path(
            args.labels_csv
            or _ROOT / "following/outputs/following_style_clusters/following_style_features.csv"
        )
        out_dir = Path(args.out_dir or labels_csv.parent)
        n_feat = 7
        weights = (
            np.array([float(x) for x in args.feature_weights.split(",")], dtype=np.float64)
            if args.feature_weights.strip()
            else np.ones(n_feat, dtype=np.float64)
        )
        run_following(data_dir, labels_csv, out_dir, weights, lambdas)
    else:
        data_dir = Path(
            args.data_dir or _ROOT / "overtaking/outputs/overtaking_phase_segments_all_p1p3"
        )
        labels_csv = Path(
            args.labels_csv
            or _ROOT / "overtaking/outputs/overtaking_style_merged/driver_overtaking_style_clusters.csv"
        )
        out_dir = Path(args.out_dir or labels_csv.parent)
        n_feat = 6
        weights = (
            np.array([float(x) for x in args.feature_weights.split(",")], dtype=np.float64)
            if args.feature_weights.strip()
            else np.ones(n_feat, dtype=np.float64)
        )
        y_thr = float(args.y_center_p2_end) if args.y_center_p2_end is not None else -3.85
        seg_kw = dict(
            right_y_min=args.right_y_min,
            right_y_max=args.right_y_max,
            ego_half_width_m=args.ego_half_width_m,
            segment_smooth_window=args.segment_smooth_window,
            edge_eps_m=args.edge_eps_m,
            center_hold_sec=args.center_hold_sec,
            y_thr_p2=y_thr,
        )
        run_overtaking(data_dir, labels_csv, out_dir, weights, lambdas, seg_kw)


if __name__ == "__main__":
    main()
