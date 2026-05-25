#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-/root/isoHC}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/isoHC/data/lm_cache}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/isoHC/results/0525_fe_fair_deep36}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/isoHC}"

TRAIN_CACHE="${TRAIN_CACHE:-${DATA_ROOT}/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt}"
VAL_CACHE="${VAL_CACHE:-${DATA_ROOT}/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt}"
TOTAL_TOKENS="${TOTAL_TOKENS:-30000000}"
MEMORY_TARGET_GB="${MEMORY_TARGET_GB:-29}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
PRESET="${PRESET:-fe-deep-36l-512}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x /root/miniconda3/bin/python3 ]]; then
    PYTHON_BIN=/root/miniconda3/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

mkdir -p "${RESULT_ROOT}" \
  "${CACHE_ROOT}/hf_cache" \
  "${CACHE_ROOT}/tmp" \
  "${CACHE_ROOT}/torchinductor_cache"

export HF_HOME="${CACHE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_cache/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf_cache/transformers"
export TMPDIR="${CACHE_ROOT}/tmp"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; refusing to launch a CPU run." >&2
  exit 1
fi
nvidia-smi >/dev/null

"${PYTHON_BIN}" -u experiments/lm_5090_next_runs.py \
  --preset "${PRESET}" \
  --methods baseline identity-hc unconstrained mhc isohc \
  --dataset fineweb-edu \
  --output_dir "${RESULT_ROOT}" \
  --total_tokens "${TOTAL_TOKENS}" \
  --train_cache_path "${TRAIN_CACHE}" \
  --val_cache_path "${VAL_CACHE}" \
  --vocab_size 50257 \
  --fair_auto_batch \
  --memory_target_gb "${MEMORY_TARGET_GB}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --compile_mode max-autotune \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --no_save_checkpoints \
  --require_cuda
