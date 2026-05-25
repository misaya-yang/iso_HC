#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-/root/isoHC}"

DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/isoHC/data/lm_cache}"
RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/isoHC/results/0525_mechanism_gpu_48l}"
CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/isoHC}"
PIPELINE_LOG="${PIPELINE_LOG:-${RESULT_ROOT}/pipeline.out}"

TRAIN_CACHE="${TRAIN_CACHE:-${DATA_ROOT}/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt}"
VAL_CACHE="${VAL_CACHE:-${DATA_ROOT}/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt}"
TOTAL_TOKENS="${TOTAL_TOKENS:-20000000}"
MEMORY_TARGET_GB="${MEMORY_TARGET_GB:-31}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-8}"
PRESET="${PRESET:-fe-deep-48l-512}"
EVAL_EVERY_TOKENS="${EVAL_EVERY_TOKENS:-1000000000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-8}"
METHODS="${METHODS:-identity-hc mhc isohc}"

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; refusing to launch a CPU/no-GPU run." >&2
  exit 1
fi
nvidia-smi >/dev/null

if [[ ! -f "${TRAIN_CACHE}" || ! -f "${VAL_CACHE}" ]]; then
  echo "Missing token caches on data disk:" >&2
  echo "  ${TRAIN_CACHE}" >&2
  echo "  ${VAL_CACHE}" >&2
  exit 1
fi

{
  echo "[$(date '+%F %T')] mechanism GPU pipeline start"
  echo "result_root=${RESULT_ROOT}"
  echo "preset=${PRESET} total_tokens=${TOTAL_TOKENS} memory_target=${MEMORY_TARGET_GB}GB"
  echo "methods=${METHODS}"

  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -u experiments/lm_5090_next_runs.py \
    --preset "${PRESET}" \
    --methods ${METHODS} \
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
    --eval_every_tokens "${EVAL_EVERY_TOKENS}" \
    --eval_max_batches "${EVAL_MAX_BATCHES}" \
    --no_save_best_checkpoints \
    --require_cuda

  echo "[$(date '+%F %T')] posthoc mechanism analysis start"
  RUN_DIRS=()
  for method in ${METHODS}; do
    RUN_DIRS+=("${RESULT_ROOT}/${PRESET}_${method}_seed0")
  done

  "${PYTHON_BIN}" -u experiments/analyze_lm_mechanisms.py \
    --run_dirs "${RUN_DIRS[@]}" \
    --output_dir "${RESULT_ROOT}/posthoc" \
    --dataset fineweb-edu \
    --val_cache_path "${VAL_CACHE}" \
    --eval_batches 4 \
    --intervention_stride 8 \
    --num_workers 2

  echo "[$(date '+%F %T')] mechanism GPU pipeline complete"
} 2>&1 | tee -a "${PIPELINE_LOG}"
