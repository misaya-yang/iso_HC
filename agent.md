# IsoHC Agent Guide

## Current Server Pattern

- Host: `connect.westd.seetacloud.com`
- User: `root`
- Port changes per cloned instance.
- Repo path: `/root/isoHC`
- Data/results path: `/root/autodl-tmp/isoHC`
- Python: `/root/miniconda3/bin/python3`

Never put token caches, HuggingFace cache, temp files, TorchInductor cache, or
large results on the system disk. Use:

```bash
export HF_HOME=/root/autodl-tmp/isoHC/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/isoHC/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/isoHC/hf_cache/transformers
export TMPDIR=/root/autodl-tmp/isoHC/tmp
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/isoHC/torchinductor_cache
```

## Current P0 Entrypoint

Use `experiments/hc_causal_controls.py` for new causal-control evidence. Start
with `--suite geometry`, then `p0-smoke`; promote only supported controls to
`p0-train` or `p0-depth`.

```text
static-birkhoff-hc is the current static Sinkhorn proxy.
It is not faithful dynamic mHC.
Do not generalize static-proxy results to official mHC without P2 parity.
```

## Historical 0525 Reproduction Command

```bash
cd /root/isoHC
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_mechanism_gpu_48l \
MEMORY_TARGET_GB=31 \
NUM_WORKERS=8 \
PREFETCH_FACTOR=8 \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

For the clean `mHC` vs `IsoHC` comparison:

```bash
METHODS="mhc isohc" \
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

## Supported Main Methods

- `identity-hc`: architecture control.
- `static-birkhoff-hc`: current Birkhoff/Sinkhorn proxy for new evidence.
- `mhc`: legacy command/checkpoint alias for that static proxy.
- `isohc`: Newton-Schulz fixed-vector isometric transport.
- `unconstrained`: unsafe oracle for drift/stability stress only.

Spectral/SVD training baselines are intentionally not in the main runner.

## Current Evidence Target

Do not frame the paper as "PPL winner". Frame it as:

```text
The static Birkhoff proxy may contract 1_perp at depth; P0 tests whether the
effect comes from initialization, training, or contraction itself.
```

Main diagnostics:

- `1_perp` singular spectrum
- composite complement gain
- mean-zero stream energy
- stream cosine / effective rank
- layer-wise stream gradient profile
- weak single-state and persistent complement interventions
- IsoHC -> identity/random-Iso replacement

## Do Not Use

The old FE overnight/fair scripts, old LM phase smoke scripts, and old
TinyShakespeare real-text grid scripts were removed. They are not valid paper
evidence for the current mechanism claim.
