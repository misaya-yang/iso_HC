# IsoHC

Canonical code for the IsoHC mechanism experiments.

## Current Main Claim

The current `static-birkhoff-hc` proxy preserves the residual-stream mean but
can contract the mean-zero stream subspace at depth. `IsoHC` keeps the same
invariant while using fixed-vector isometric transport on the complement. This
is a hypothesis under causal testing, not a claim about faithful dynamic mHC.

## Supported Experiment Entrypoints

- `experiments/hc_causal_controls.py`
  - Current P0 geometry, matched-control, depth, and intervention suites.
- `experiments/run_0525_mechanism_gpu_pipeline.sh`
  - Historical 5090 FE 48L reproduction run.
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

`mhc` remains a legacy command/checkpoint alias. New evidence must use the
label `static-birkhoff-hc`.

## P0 Geometry Command

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/hc_causal_controls.py \
  --suite geometry \
  --output_dir outputs/hc_geometry
```

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
