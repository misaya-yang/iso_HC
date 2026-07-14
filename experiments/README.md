# Experiments

## P0 Causal Controls

Offline CPU orchestration check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/hc_causal_controls.py \
  --suite p0-smoke \
  --output_dir outputs/hc_p0_smoke \
  --dataset random \
  --total_tokens 4096 \
  --batch_size 2 \
  --no_compile
```

Server P0 run using existing data-disk caches:

```bash
/root/miniconda3/bin/python3 -u experiments/hc_causal_controls.py \
  --suite p0-train \
  --output_dir /root/autodl-tmp/isoHC/results/p0_causal_seed0 \
  --dataset fineweb-edu \
  --total_tokens 20000000 \
  --batch_size 20 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt
```

`p0-depth` overrides the global batch per depth to stay below the calibrated
28 GB limit: 24/48/72/96/128 layers use batch 29/20/16/14/12 respectively.

The runner refuses completed output directories. Use a new output root for a
rerun. `static-birkhoff-hc` is a static Sinkhorn proxy, not faithful dynamic
mHC; `mhc` is retained only as a legacy alias.

## Canonical LM Pipeline

Use this for the current paper line:

```bash
cd /root/isoHC
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_mechanism_gpu_48l \
MEMORY_TARGET_GB=31 \
NUM_WORKERS=8 \
PREFETCH_FACTOR=8 \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

Default methods:

```text
identity-hc mhc isohc
```

Targeted core rerun:

```bash
METHODS="mhc isohc" \
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

## Current Files

- `lm_5090_next_runs.py`: 5090 FE/LM runner.
- `hc_causal_controls.py`: P0 causal experiment matrix and safe orchestration.
- `run_0525_mechanism_gpu_pipeline.sh`: safe server pipeline with data-disk
  cache paths, fair batch probing, checkpointed runs, and posthoc analysis.
- `analyze_lm_mechanisms.py`: checkpoint posthoc analysis for gradient,
  complement gain, weak single-state removal, persistent complement scaling,
  and replacement interventions.
- `prepare_lm_data.py`: token cache preparation.
- `stage1_projection_sanity.py`, `stage1_residual_only.py`,
  `stage2_precision_depth_suite.py`, `stage2_stability_detectors.py`: mechanism
  diagnostics and sanity checks.
- `gnn_*`: earlier graph-side exploratory diagnostics, kept separate from the
  current LM claim.

## Removed Legacy Entrypoints

The following old scripts were removed because they encouraged wrong or stale
runs:

- `run_0525_fe_fair_deep.sh`
- `run_0525_fe_overnight_pipeline.sh`
- `lm_phase0_smoke.py`
- `lm_phase1_controlled.py`
- `lm_verify.py`
- `stage2_real_text_smoke.py`
- `stage2_real_text_grid.py`

Spectral/SVD training baselines are not part of the main runner. IsoHC training
uses Newton-Schulz fixed-vector projection; SVD is reserved for diagnostics or
exact sanity checks.
