#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR:-/root/isoHC}"

RESULT_ROOT="${RESULT_ROOT:-/root/autodl-tmp/isoHC/results/0525_fe_fair_deep48_p33013_20m}"
PIPELINE_LOG="${PIPELINE_LOG:-${RESULT_ROOT}/pipeline.out}"
mkdir -p "${RESULT_ROOT}"

{
  echo "[$(date '+%F %T')] overnight FE 48L pipeline start"
  echo "result_root=${RESULT_ROOT}"

  PRESET=fe-deep-48l-512 \
  RESULT_ROOT="${RESULT_ROOT}" \
  TOTAL_TOKENS="${TOTAL_TOKENS:-20000000}" \
  MEMORY_TARGET_GB="${MEMORY_TARGET_GB:-30}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}" \
    bash experiments/run_0525_fe_fair_deep.sh

  echo "[$(date '+%F %T')] training finished; writing summary"
  /root/miniconda3/bin/python3 - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["RESULT_ROOT"])
rows = []
for path in sorted(root.glob("*/run_summary.json")):
    with path.open() as f:
        data = json.load(f)
    cfg = data.get("config", {})
    method = cfg.get("method", path.parent.name)
    final_eval = data.get("final_eval", {})
    train_metrics = data.get("train_metrics", [])
    posthoc = data.get("posthoc", {})
    last_train = train_metrics[-1] if train_metrics else {}
    rows.append({
        "method": method,
        "success": data.get("success"),
        "batch": cfg.get("batch_size"),
        "tokens_m": last_train.get("total_tokens", 0) / 1e6,
        "tok_s": last_train.get("tokens_per_sec"),
        "elapsed_min": data.get("elapsed_sec", 0) / 60,
        "val_loss": final_eval.get("val_loss"),
        "val_ppl": final_eval.get("val_ppl"),
        "e0": posthoc.get("mean_zero_energy_initial"),
        "e1": posthoc.get("mean_zero_energy_final"),
        "c0": posthoc.get("stream_cosine_initial"),
        "c1": posthoc.get("stream_cosine_final"),
    })

probe = next(root.glob("*_fair_batch_probe.json"), None)
lines = [
    "# 0525 FE 48L Overnight Summary",
    "",
    f"Result root: `{root}`",
    "",
]
if probe:
    lines += ["## Fair Batch Probe", "", "```json", probe.read_text().strip(), "```", ""]

lines += [
    "## Runs",
    "",
    "| method | success | batch | tokens(M) | tok/s | elapsed(min) | val_loss | val_ppl | E_perp init->final | cosine init->final |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
]
for r in rows:
    def fmt(x, nd=4):
        if x is None:
            return ""
        if isinstance(x, float) and not math.isfinite(x):
            return "nan"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    lines.append(
        f"| {r['method']} | {r['success']} | {r['batch']} | {fmt(r['tokens_m'], 1)} | "
        f"{fmt(r['tok_s'], 0)} | {fmt(r['elapsed_min'], 1)} | {fmt(r['val_loss'], 4)} | "
        f"{fmt(r['val_ppl'], 2)} | {fmt(r['e0'], 4)} -> {fmt(r['e1'], 4)} | "
        f"{fmt(r['c0'], 4)} -> {fmt(r['c1'], 4)} |"
    )

(root / "overnight_summary.md").write_text("\n".join(lines) + "\n")
print(root / "overnight_summary.md")
PY

  echo "[$(date '+%F %T')] pipeline complete"
} 2>&1 | tee -a "${PIPELINE_LOG}"
