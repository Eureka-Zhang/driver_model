#!/bin/bash
# Parallel per-driver training across multiple GPUs.
# Splits drivers round-robin over GPUs and launches training in parallel.
#
# Usage:
#   bash following/gru_train/run_all_v3_parallel.sh                    # use GPUs 0-3
#   NUM_GPUS=2 bash following/gru_train/run_all_v3_parallel.sh         # use GPUs 0-1
#   GPUS="0,2" bash following/gru_train/run_all_v3_parallel.sh         # explicit list
#   bash following/gru_train/run_all_v3_parallel.sh --epochs 30        # extra args
#
set -u

ROOT="$(cd "$(dirname "$0")"/../../ && pwd)"
CAL_DIR="${ROOT}/following/outputs/following_calibrated"
IDM_DIR="${ROOT}/following/outputs/idm_per_driver"
OUT_ROOT="${ROOT}/following/outputs/residual_gru_v3"
TRAIN_SCRIPT="${ROOT}/following/gru_train/train_bc_gru_residual_v3.py"

# GPU list: explicit GPUS="0,2" or NUM_GPUS=N (default 4)
if [ -n "${GPUS:-}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
else
  N="${NUM_GPUS:-4}"
  GPU_IDS=()
  for ((i=0; i<N; i++)); do GPU_IDS+=("$i"); done
fi

# Pick python with torch
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
[ -z "${PYTHON_BIN}" ] && { echo "[ERR] no python with torch" >&2; exit 2; }

# Parse extra args and optional explicit driver list
EXPLICIT_DRIVERS=()
EXTRA_ARGS=()
for a in "$@"; do
  if [[ "$a" =~ ^T[0-9]+$ ]]; then EXPLICIT_DRIVERS+=("$a")
  else EXTRA_ARGS+=("$a"); fi
done

if [ "${#EXPLICIT_DRIVERS[@]}" -eq 0 ]; then
  DRIVERS=()
  for dir in "${CAL_DIR}"/T*/; do
    d="$(basename "$dir")"
    [ -f "${IDM_DIR}/${d}/idm.json" ] && DRIVERS+=("$d")
  done
else
  DRIVERS=("${EXPLICIT_DRIVERS[@]}")
fi
IFS=$'\n' DRIVERS=($(printf "%s\n" "${DRIVERS[@]}" | sort -t T -k2 -n))
unset IFS

mkdir -p "${OUT_ROOT}" "${OUT_ROOT}/_logs"

echo "== Residual GRU v3 parallel training =="
echo "Python : ${PYTHON_BIN}"
echo "GPUs   : ${GPU_IDS[*]}"
echo "Drivers: ${DRIVERS[*]}"
echo "Extra  : ${EXTRA_ARGS[*]:-<none>}"
echo

# Launch: round-robin drivers over GPUs, keep only NUM_GPUS jobs in flight.
NUM_GPUS_USED=${#GPU_IDS[@]}
PIDS=()
GPU_OF=()

launch_on_gpu() {
  local d="$1" gpu="$2"
  local out="${OUT_ROOT}/${d}" log="${OUT_ROOT}/_logs/${d}.log"
  mkdir -p "${out}"
  echo "[launch] GPU${gpu} -> ${d} (log: ${log})"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}" \
    --data_dir "${CAL_DIR}" \
    --idm_dir  "${IDM_DIR}" \
    --out_dir  "${out}" \
    --drivers  "${d}" \
    --device cuda \
    "${EXTRA_ARGS[@]}" \
    >"${log}" 2>&1 &
  PIDS+=("$!")
  GPU_OF+=("${gpu}")
}

i=0
active_on_gpu() {
  local gpu="$1" count=0
  for idx in "${!PIDS[@]}"; do
    local pid="${PIDS[$idx]}"
    kill -0 "${pid}" 2>/dev/null && [ "${GPU_OF[$idx]}" = "${gpu}" ] && count=$((count+1))
  done
  echo "${count}"
}

for d in "${DRIVERS[@]}"; do
  # Find first free GPU
  while : ; do
    for gpu in "${GPU_IDS[@]}"; do
      if [ "$(active_on_gpu "${gpu}")" = "0" ]; then
        launch_on_gpu "${d}" "${gpu}"
        break 2
      fi
    done
    sleep 3
  done
done

echo
echo "All launched. Waiting for completion..."
FAIL=()
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if ! wait "${pid}"; then
    FAIL+=("pid=${pid} gpu=${GPU_OF[$idx]}")
  fi
done

echo "== DONE =="
echo "Trained $(( ${#DRIVERS[@]} - ${#FAIL[@]} )) / ${#DRIVERS[@]} drivers"
[ "${#FAIL[@]}" -gt 0 ] && { echo "Failed: ${FAIL[*]}"; exit 1; }
