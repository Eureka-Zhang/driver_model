# Residual GRU car-following pipeline (v3)

End-to-end flow for training a **gain + residual GRU** on top of a per-driver **IDM**, then generating **closed-loop takeover** CSVs where the policy controls ego longitudinal dynamics after a chosen time while the **lead trajectory stays real**.

| Stage | Script / module | Role |
|--------|------------------|------|
| 1. Train | `train_bc_gru_residual_v3.py` | Behaviour cloning with short rollouts + state anchoring; writes `best_model.pt` and metadata. |
| 2. Inference API | `residual_gru_policy.py` | Loads weights and runs `a_pred = (1 + α) · a_IDM + Δa` step-by-step. |
| 3. Takeover data | `generate_residual_gru_takeover.py` | Keeps first *N* seconds of real data per session, then rolls out the policy. |

Shared utilities: `bc_gru_features.py` (row aliases, `inv_ttc` / `inv_time_headway` helpers) is imported by the trainer.

---

## Model idea (v3)

Longitudinal acceleration is predicted as:

```text
a_pred = clip( (1 + α) · a_IDM(v, gap, lead_v) + Δa )
```

- The **GRU** outputs bounded **α** (IDM gain rescale) and **Δa** (residual).
- **IDM** parameters come from the same driver’s `idm.json` (calibrated IDM).
- Training uses **short sub-segments** (default 25 frames) and **anchoring**: closed-loop `(v_cf, gap_cf)` is softly pulled toward the logged `(v_real, gap_real)` each step so features do not drift away from what the teacher saw. At inference, **no anchoring** is applied.

Details and hyperparameter intuition are in the docstring at the top of `train_bc_gru_residual_v3.py`.

---

## Prerequisites

- **Python 3** with **PyTorch** and **NumPy**.
- **Input data**
  - Calibrated following logs: e.g. `following/outputs/following_calibrated/**/driving_data.csv` (and optionally `segment_*.csv` under the same tree).
  - Per-driver IDM fits: `following/outputs/idm_per_driver/<T*>/idm.json`.

---

## 1. Train (`train_bc_gru_residual_v3.py`)

Trains one model per `--out_dir` (typically one directory per driver, e.g. `.../residual_gru_v3/T5`).

**Outputs** (under `--out_dir`):

| File | Content |
|------|---------|
| `best_model.pt` | GRU weights (best validation). |
| `model_meta.json` | Architecture, `seq_len`, feature order, IDM paths, driver list, etc. |
| `train_report.json` | Training metrics, **feature mean/std** used for normalization at inference. |

**Example:**

```bash
python3 following/gru_train/train_bc_gru_residual_v3.py \
  --data_dir /path/to/following/outputs/following_calibrated \
  --idm_dir /path/to/following/outputs/idm_per_driver \
  --out_dir /path/to/following/outputs/residual_gru_v3/T5 \
  --drivers T5 \
  --seq_len 20 --segment_len 25 --segment_stride 15 \
  --epochs 60 \
  --anchor_start 0.5 --anchor_floor 0.1 --anchor_anneal_epochs 25 \
  --min_sim_time_s 15
```

**Useful flags:**

- `--drivers` — comma-separated `T1,T2,...`; omit to train all drivers that have both CSVs and `idm.json`.
- `--split_mode` — `per_file` (default) vs `per_subsegment` when a driver has very few files.
- `--device` — `auto` | `cuda` | `cpu`.

Repeat with different `--out_dir` / `--drivers` for each driver you need in the next step (the takeover script expects `<model_root>/<T*>/` per driver).

---

## 2. Policy module (`residual_gru_policy.py`)

Not a standalone CLI; it is imported by `generate_residual_gru_takeover.py` (and can be used from other code).

**`ResidualGRUPolicy(model_dir, device=...)`** expects:

- `best_model.pt`, `model_meta.json`, `train_report.json` in `model_dir`.
- `arch` in meta must be `gain_residual_gru_v3`.
- IDM defaults: `model_meta.json` stores `idm_dir`; policy loads `<idm_dir>/<driver>/idm.json` unless you pass `idm_params` / `driver_id`.

**Typical pattern:**

1. `hist = policy.init_history(...)` with **last `seq_len`** real samples (7 features per row, order in `FEATURE_ORDER`).
2. `a_pred, info = policy.step(v_cur, gap_cur, lead_v, hist)`.
3. Integrate `v`, `gap`; `hist = policy.push_history(...)`.

Optional helper: `rollout_against_lead(...)` for a full simulation given a lead speed sequence.

---

## 3. Generate takeover CSVs (`generate_residual_gru_takeover.py`)

For each driver:

1. Chooses one **calibrated** session (`driving_data.csv` under `--calibrated_dir`).
2. Copies all rows **before** `sim_time_s >= takeover_time_s` unchanged.
3. From the takeover row onward, runs **closed-loop** longitudinal update with `ResidualGRUPolicy`: real **lead speed** per row, predicted **ego acceleration / speed / gap**; lateral columns stay from the original file.
4. Appends diagnostic columns when missing: `gru_alpha`, `gru_delta_a`, `a_idm_base`, etc.

**Outputs:**

- `<out_dir>/<T*>/driving_data.csv` — full-length CSV, mixed real + rolled-out tail.
- `<out_dir>/generation_summary.csv` — one row per successful driver (paths, takeover index, row counts).

**Example:**

```bash
python3 following/gru_train/generate_residual_gru_takeover.py \
  --calibrated_dir /path/to/following/outputs/following_calibrated \
  --model_root /path/to/following/outputs/residual_gru_v3 \
  --out_dir /path/to/following/outputs/residual_gru_takeover_20s \
  --takeover_time_s 20.0 \
  --session_index -1 \
  --device cpu
```

**Important flags:**

| Flag | Meaning |
|------|---------|
| `--model_root` | Directory that contains **per-driver** subfolders `T1`, `T2`, … each with the three training artifacts. |
| `--takeover_time_s` | First row where `sim_time_s >=` this value is controlled by the GRU+IDM. |
| `--session_index` | **0-based** session index after sorting paths per driver. **Default `-1` = last session** (not first). Use `0` for the first session. |
| `--drivers` | Subset of drivers; empty = all discovered. |

**Requirements per driver:** at least `seq_len` rows of history before the takeover row; otherwise the driver is skipped with a message.

---

## Directory layout (recommended)

```text
following/outputs/
  following_calibrated/     # teacher CSVs
  idm_per_driver/           # <T*>/idm.json
  residual_gru_v3/          # model_root
    T1/  best_model.pt, model_meta.json, train_report.json
    T2/
    ...
  residual_gru_takeover_20s/   # generate script out_dir
    T1/driving_data.csv
    ...
    generation_summary.csv
```

---

## Troubleshooting

- **`[SKIP] incomplete model`** — Train that driver first, or fix `--model_root` / folder name (`T*` must match).
- **`takeover at row < seq_len`** — Use a longer session, smaller `--takeover_time_s`, or train with a smaller `--seq_len` (must match at inference).
- **Feature / arch errors** — Do not mix checkpoints from a different trainer version; `residual_gru_policy` checks `arch == gain_residual_gru_v3` and feature order.

---

## See also

- `cluster_following_style.py` / calibrated data prep elsewhere in `following/scripts/` for upstream data.
- Docstrings in each file for full argument lists and training loss details.
