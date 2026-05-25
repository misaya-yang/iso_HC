# Experiments

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
- `run_0525_mechanism_gpu_pipeline.sh`: safe server pipeline with data-disk
  cache paths, fair batch probing, checkpointed runs, and posthoc analysis.
- `analyze_lm_mechanisms.py`: checkpoint posthoc analysis for gradient,
  complement gain, complement removal, and replacement interventions.
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
