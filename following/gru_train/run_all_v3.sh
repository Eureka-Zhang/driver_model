#!/bin/bash
# -*- coding: utf-8 -*-
#
# Train per-driver gain+residual GRU (v3) for every T<n> that has:
#   1. calibrated driving CSVs under <CAL_DIR>/<T*>/
#   2. fitted IDM params at <IDM_DIR>/<T*>/idm.json
#
# Trains each driver independently and writes:
#   <OUT_ROOT>/<T*>/best_model.pt
#   <OUT_ROOT>/<T*>/model_meta.json
#   <OUT_ROOT>/<T*>/train_report.json
#
# Usage:
#   bash following/gru_train/run_all_v3.sh                 # run every driver
#   bash following/gru_train/run_all_v3.sh T1 T5 T10       # restrict to given list
#   bash following/gru_train/run_all_v3.sh --epochs 40     # pass extra args to v3
#

set -u

ROOT="$(cd "$(dirname "$0")"/../../ && pwd)"
CAL_DIR="${ROOT}/following/outputs/following_calibrated"
IDM_DIR="${ROOT}/following/outputs/idm_per_driver"
OUT_ROOT="${ROOT}/following/outputs/residual_gru_v3"
TRAIN_SCRIPT="${ROOT}/following/gru_train/train_bc_gru_residual_v3.py"

# Pick Python interpreter: prefer an env with torch installed. Override by
# setting PYTHON_BIN=/path/to/python in the environment.
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  for cand in \
    /home/zwx/miniconda3/envs/carla/bin/python \
    python3 \
    python; do
    if command -v "${cand}" >/dev/null 2>&1 && \
       "${cand}" -c "import torch" >/dev/null 2>&1; then
      PYTHON_BIN="${cand}"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "[ERR] no Python interpreter with torch found. Set PYTHON_BIN=/path/to/python." >&2
  exit 2
fi
echo "Python  : ${PYTHON_BIN}  ($(${PYTHON_BIN} -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())'))"

# Parse args: explicit driver list vs extra v3 flags.
EXPLICIT_DRIVERS=()
EXTRA_ARGS=()
for a in "$@"; do
  if [[ "$a" =~ ^T[0-9]+$ ]]; then
    EXPLICIT_DRIVERS+=("$a")
  else
    EXTRA_ARGS+=("$a")
  fi
done

# Discover drivers that have both calibrated data and an IDM fit.
if [ "${#EXPLICIT_DRIVERS[@]}" -eq 0 ]; then
  DRIVERS=()
  for dir in "${CAL_DIR}"/T*/; do
    d="$(basename "$dir")"
    if [ -f "${IDM_DIR}/${d}/idm.json" ]; then
      DRIVERS+=("$d")
    else
      echo "[SKIP] ${d}: no IDM fit at ${IDM_DIR}/${d}/idm.json"
    fi
  done
else
  DRIVERS=("${EXPLICIT_DRIVERS[@]}")
fi

# Sort drivers by numeric suffix (T1, T2, ..., T20).
IFS=$'\n' DRIVERS=($(printf "%s\n" "${DRIVERS[@]}" | sort -t T -k2 -n))
unset IFS

echo "== Residual GRU v3 batch training =="
echo "Drivers : ${DRIVERS[*]}"
echo "Out root: ${OUT_ROOT}"
echo "Extra   : ${EXTRA_ARGS[*]:-<none>}"
echo

mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/_logs"
mkdir -p "${LOG_DIR}"

FAIL=()
for d in "${DRIVERS[@]}"; do
  OUT_DIR="${OUT_ROOT}/${d}"
  LOG="${LOG_DIR}/${d}.log"
  echo "---- [${d}] ----  -> ${OUT_DIR}"
  mkdir -p "${OUT_DIR}"

  python_bin_safe="${PYTHON_BIN}"
  "${python_bin_safe}" -u "${TRAIN_SCRIPT}" \
    --data_dir "${CAL_DIR}" \
    --idm_dir  "${IDM_DIR}" \
    --out_dir  "${OUT_DIR}" \
    --drivers  "${d}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG}"

  STATUS="${PIPESTATUS[0]}"
  if [ "${STATUS}" -ne 0 ]; then
    echo "[FAIL] ${d} (exit ${STATUS}) — see ${LOG}"
    FAIL+=("${d}")
  fi
  echo
done

echo "== DONE =="
echo "Trained: $(( ${#DRIVERS[@]} - ${#FAIL[@]} )) / ${#DRIVERS[@]}"
if [ "${#FAIL[@]}" -gt 0 ]; then
  echo "Failed : ${FAIL[*]}"
  exit 1
fi
