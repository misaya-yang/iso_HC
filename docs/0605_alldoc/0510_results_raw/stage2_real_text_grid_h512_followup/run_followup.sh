#!/usr/bin/env bash
set -o pipefail
run_one() {
  model="$1"; layers="$2"; batch="$3"
  echo "======================================================================"
  echo "Followup real text: model=${model} layers=${layers} batch=${batch}"
  echo "======================================================================"
  /root/miniconda3/bin/python3 experiments/stage2_real_text_smoke.py \
    --model "${model}" \
    --layers "${layers}" \
    --hidden-dim 512 \
    --heads 8 \
    --streams 8 \
    --context-length 512 \
    --steps 2000 \
    --batch-size "${batch}" \
    --precision bf16 \
    --learning-rate 3e-4 \
    --warmup-steps 200 \
    --device cuda \
    --output-dir results/stage2_real_text_grid_h512_followup \
    --data-path data/tinyshakespeare/input.txt \
    2>&1 | tee "results/stage2_real_text_grid_h512_followup/${model}_L${layers}_B${batch}.log"
  status=${PIPESTATUS[0]}
  echo "RUN_STATUS model=${model} layers=${layers} batch=${batch} status=${status}" | tee -a results/stage2_real_text_grid_h512_followup/run_status.log
  return "${status}"
}
: > results/stage2_real_text_grid_h512_followup/run_status.log
run_one isohc 16 32 || exit $?
run_one baseline 32 16 || exit $?
run_one isohc 32 16 || exit $?
