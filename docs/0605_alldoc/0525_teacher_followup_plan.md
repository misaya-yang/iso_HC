# 0525 Teacher Follow-up Plan

## Current Position

The 48L FineWeb-Edu fair run already supports the mechanism claim:

```text
mHC:   mean-zero singular value ~= 0.922, E_perp 0.0224 -> 0.0136
IsoHC: mean-zero singular value ~= 1.000, E_perp 0.0224 -> 0.0546
```

The safe paper claim is no longer "IsoHC is a PPL winner". It should be:

```text
Birkhoff/mHC preserves the residual mean and prevents explosion, but it can
introduce depth-wise contraction on 1_perp. IsoHC is the fixed-vector isometric
repair for that missing geometry.
```

The missing evidence is whether the preserved `1_perp` signal is actually used
by the model, and whether identity-HC / unconstrained HC / spectral baselines
explain away IsoHC.

## Code Added For This Phase

- `lm/transport_analysis.py`
  - per-layer `1_perp` singular spectrum
  - cumulative/composite complement gain
  - product-of-single-step contraction curve
- `lm/mixing.py`
  - `spectral`
  - `fixed-vector-spectral`
- `lm/models.py`
  - validation-time stream intervention: `mean_only` / `scale_perp`
  - stream-state gradient capture
  - mixer override: identity or supplied matrices
  - ordered `get_named_mixing_matrices()`
- `experiments/analyze_lm_mechanisms.py`
  - checkpoint posthoc analysis for composite gain, gradient profile,
    complement removal, IsoHC->I, and IsoHC->random-Iso.
- `experiments/lm_5090_next_runs.py`
  - supports `spectral-hc` and `fixed-vector-spectral-hc`
  - stores per-layer mixer diagnostics and transport complement report.

## Fairness Rules

All comparison runs must keep:

- same FE token cache and validation cache
- same tokenizer, context length, token budget, seed, LR schedule, warmup, weight decay, grad clip
- same `TwoBranchHCTransformer` architecture for HC methods
- same fair common batch selected by `--fair_auto_batch`
- same checkpoint policy inside a comparison group
- data, HF cache, temp, and TorchInductor cache on `/root/autodl-tmp/isoHC`, not system disk

Use `identity-hc` as the architecture control. Use `mHC vs IsoHC` as the cleanest geometry comparison. Treat unconstrained HC as a flexible but unsafe oracle unless it passes drift/composite/seed/depth checks.

## China Server Data Rule

The server may not have reliable HuggingFace access. Do not depend on live HF during training.

Preferred path:

```text
/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt
/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt
```

If the cache must be rebuilt on the server, use a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/isoHC/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/isoHC/hf_cache/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/isoHC/hf_cache/transformers
export TMPDIR=/root/autodl-tmp/isoHC/tmp
```

If built locally, upload only the final `.pt` token caches to the data disk and delete local temporary parquet/cache files after upload.

## Step 1: Checkpointed Mechanism Runs

Reason: the previous overnight run used `--no_save_checkpoints`, so it proves the aggregate mechanism but cannot run complement-removal or replacement interventions.

Recommended GPU pipeline after the server is opened with a card:

```bash
cd /root/isoHC
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_mechanism_gpu_48l \
MEMORY_TARGET_GB=31 \
NUM_WORKERS=8 \
PREFETCH_FACTOR=8 \
bash experiments/run_0525_mechanism_gpu_pipeline.sh
```

This pipeline uses fair batch probing with finer candidates, keeps all caches on
`/root/autodl-tmp/isoHC`, disables repeated `best.pt` writes, lowers mid-training
evaluation frequency, saves final checkpoints, then runs posthoc analysis.

Run a checkpointed 48L mechanism set:

```bash
cd /root/isoHC
RESULT_ROOT=/root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l \
PRESET=fe-deep-48l-512 \
TOTAL_TOKENS=20000000 \
MEMORY_TARGET_GB=30 \
NUM_WORKERS=4 \
PREFETCH_FACTOR=4 \
bash experiments/run_0525_fe_fair_deep.sh
```

For this checkpointed run, remove `--no_save_checkpoints` from the shell script or run the Python command directly with:

```bash
/root/miniconda3/bin/python3 -u experiments/lm_5090_next_runs.py \
  --preset fe-deep-48l-512 \
  --methods identity-hc mhc isohc fixed-vector-spectral-hc \
  --dataset fineweb-edu \
  --output_dir /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l \
  --total_tokens 20000000 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt \
  --vocab_size 50257 \
  --fair_auto_batch \
  --memory_target_gb 30 \
  --compile_mode max-autotune \
  --num_workers 4 \
  --prefetch_factor 4 \
  --require_cuda
```

Expected time on 5090: roughly 60-90 minutes for four 48L methods at 20M tokens, plus compile/probe overhead. Disk cost is several GB for checkpoints, on data disk only.

## Step 2: Posthoc Mechanism Analysis

After Step 1:

```bash
/root/miniconda3/bin/python3 -u experiments/analyze_lm_mechanisms.py \
  --run_dirs \
    /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l/fe-deep-48l-512_identity-hc_seed0 \
    /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l/fe-deep-48l-512_mhc_seed0 \
    /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l/fe-deep-48l-512_isohc_seed0 \
    /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l/fe-deep-48l-512_fixed-vector-spectral-hc_seed0 \
  --output_dir /root/autodl-tmp/isoHC/results/0525_mech_ckpt_48l/posthoc \
  --dataset fineweb-edu \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt \
  --eval_batches 4 \
  --intervention_stride 8
```

Primary figures/tables from this step:

- composite `1_perp` gain: mHC should decay, IsoHC should stay near 1
- gradient profile slope: mHC should show stronger attenuation if contraction matters
- complement removal: IsoHC should have larger `delta_loss` if its preserved complement is useful
- IsoHC->identity: positive `delta_loss` means learned rotation matters
- fixed-vector spectral: tests whether "non-expansive" is enough without isometry

## Step 3: Seed Robustness

Run 48L 20M tokens for two more seeds:

```bash
for SEED in 1 2; do
  /root/miniconda3/bin/python3 -u experiments/lm_5090_next_runs.py \
    --preset fe-deep-48l-512 \
    --methods identity-hc unconstrained mhc isohc fixed-vector-spectral-hc \
    --dataset fineweb-edu \
    --output_dir /root/autodl-tmp/isoHC/results/0525_fe48_seeds \
    --total_tokens 20000000 \
    --seed "${SEED}" \
    --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt \
    --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt \
    --vocab_size 50257 \
    --fair_auto_batch \
    --memory_target_gb 30 \
    --compile_mode max-autotune \
    --num_workers 4 \
    --prefetch_factor 4 \
    --no_save_checkpoints \
    --require_cuda
done
```

This gives mean/std for `E_perp`, `sv_1perp`, val loss, and unconstrained drift.

## Step 4: Depth Scaling

Depth matters more than token count for the contraction claim.

Run:

```text
L in {24, 48, 72}
methods: identity-hc, mhc, isohc, fixed-vector-spectral-hc, unconstrained
tokens: 20M
```

Success criterion:

```text
mHC composite/product complement gain decays faster as L grows,
IsoHC stays near 1,
and the loss/intervention gap does not contradict the mechanism.
```

## Step 5: 100M Token Bridge

Only after Step 2/3 show useful complement signal:

```bash
/root/miniconda3/bin/python3 -u experiments/lm_5090_next_runs.py \
  --preset fe-deep-48l-512 \
  --methods mhc isohc \
  --dataset fineweb-edu \
  --output_dir /root/autodl-tmp/isoHC/results/0525_fe48_mhc_vs_iso_100m \
  --total_tokens 100000000 \
  --train_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt \
  --val_cache_path /root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt \
  --vocab_size 50257 \
  --fair_auto_batch \
  --memory_target_gb 30 \
  --compile_mode max-autotune \
  --num_workers 4 \
  --prefetch_factor 4 \
  --no_save_checkpoints \
  --require_cuda
```

This is the outcome bridge. If PPL gap still stays small but intervention/gradient/composite results are strong, write the paper as a mechanism paper. If the loss gap grows, upgrade the claim.

## Stop Conditions

Do not spend expensive GPU time on 100M+ if:

- IsoHC->identity has near-zero `delta_loss`
- complement removal has near-zero `delta_loss`
- fixed-vector spectral matches IsoHC on all mechanism and outcome metrics

In that case the correct claim is narrower:

```text
Birkhoff contraction is real; fixed-vector isometry is the correct safe family,
but learned IsoHC mixing is not yet shown to improve LM outcomes at this scale.
```
