# IsoHC

Canonical code for the IsoHC mechanism experiments.

## Current Main Claim

`mHC/Birkhoff` preserves the residual-stream mean but can contract the
mean-zero stream subspace at depth. `IsoHC` keeps the same invariant while using
Newton-Schulz fixed-vector isometric transport on the complement.

## Supported Experiment Entrypoints

- `experiments/run_0525_mechanism_gpu_pipeline.sh`
  - 5090 FE 48L mechanism run.
  - Default methods: `identity-hc mhc isohc`.
  - Data/cache/results must live under `/root/autodl-tmp/isoHC`.
- `experiments/lm_5090_next_runs.py`
  - Low-level runner used by the shell pipeline.
- `experiments/analyze_lm_mechanisms.py`
  - Checkpoint posthoc analysis: composite complement gain, gradient profile,
    complement removal, IsoHC->identity, IsoHC->random-Iso.
- `experiments/prepare_lm_data.py`
  - Builds token caches when network access is available.

Do not use deleted legacy FE or TinyShakespeare/PPL smoke scripts for paper
evidence. Historical raw results are kept under `docs/0605_alldoc`.

## Server Command

```bash
cd /root/isoHC
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_mechanism_gpu_48l \
MEMORY_TARGET_GB=31 \
NUM_WORKERS=8 \
PREFETCH_FACTOR=8 \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

For the clean core comparison only:

```bash
METHODS="mhc isohc" \
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```
