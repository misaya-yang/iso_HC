# HC Causal Experiments Design

**Status:** Approved design direction; implementation awaits review of this written specification.

**Source:** [`docs/0714_experiment_guidance.md`](../../0714_experiment_guidance.md)

## Goal

Turn the supplied guidance into falsifiable experiments without treating its claims as established facts. The first implementation answers the current static-model causal questions using the repository's existing LM runner; faithful dynamic mHC, new BiLip-HC architecture, synthetic necessity tasks, and scaling runs remain explicit follow-on projects rather than being approximated inside this patch.

## Evidence Policy

- The current `MHCMixing` implementation is reported as `static-birkhoff-hc`. The legacy `mhc` name remains loadable only for existing commands and checkpoints.
- A configuration, theorem-inspired control, or diagnostic is a hypothesis until a recorded run supports it.
- Initial geometry, training-time geometry, final geometry, and intervention results are separate evidence fields; none substitutes for another.
- Noise-free analytic predictions are checked against measured matrices. Noisy or trained matrices are reported from measurements, not inferred from initialization formulas.
- Failed, non-finite, interrupted, and unavailable-GPU runs remain visible as such. A local CPU smoke test is not evidence for a 5090/FineWeb result.
- The guidance may be revised or rejected when code, source papers, or experiments disagree with it.

## Scope Decomposition

The source guidance spans four independently reviewable projects. Combining them would make a failed experiment impossible to diagnose.

| Stage | Question | Deliverable | Gate to next stage |
| --- | --- | --- | --- |
| P0: causal controls | Is the observed contraction caused by static Birkhoff initialization, training, or contraction itself? | Configurable static controls, exact geometry trajectory, persistent complement intervention | Reproducible geometry and intervention artifacts with matched configs |
| P1: accessibility | Is complement state controllable, observable, and functionally used? | Gate scans, Gramians, rotating-routing and delayed-copy tasks | Multi-seed functional effect beyond transport preservation |
| P2: method fidelity | Does faithful dynamic mHC behave like the static proxy, and is budgeted BiLip-HC useful? | Source-pinned dynamic mHC parity implementation, a separate BiLip-HC spec, and minimal Givens/Householder IsoHC | Parity tests plus matched-cost evidence |
| P3: scaling and systems | Does the surviving mechanism scale at acceptable cost? | 100M–1B token-budget matrix and kernel/system profiling | Statistical and wall-clock evidence, not a single run |

This specification fully defines P0. P1–P3 retain the full requested direction but each requires its own design review after P0 evidence exists.

## P0 Architecture

### Reuse the canonical path

The implementation reuses `experiments/lm_5090_next_runs.py`, `TwoBranchHCTransformer`, `run_experiment`, `collect_transport_report`, and the existing posthoc analyzer. It adds one focused experiment-matrix entrypoint; it does not create a second trainer, checkpoint format, or data loader.

```text
experiments/hc_causal_controls.py
  -> build_preset_configs(...)
  -> config with variant and mixer controls
  -> create_model(...)
  -> TwoBranchHCTransformer(..., mixing_kwargs=...)
  -> existing run_experiment(...)
  -> run_summary.json + posthoc intervention JSON
```

### Configuration contract

Every causal run stores these values verbatim in `run_summary.json`:

```python
{
    "method": "static-birkhoff-hc",
    "experiment_variant": "birkhoff_d4_t1_noise0_trainable",
    "mixing_kwargs": {
        "diag_bias": 4.0,
        "temperature": 1.0,
        "noise_std": 0.0,
        "sinkhorn_iters": 10,
        "identity_blend": 1.0,
    },
    "lambda_a": 0.01,
    "lambda_b": 0.01,
    "freeze_mixing": False,
    "metric_schema_version": 2,
}
```

`TwoBranchHCTransformer` accepts an optional `mixing_kwargs` mapping while retaining its existing explicit defaults. `create_model` reads the configuration instead of hard-coding `lambda_a`, `lambda_b`, and mixer settings. When `freeze_mixing` is true, only parameters owned by `attn_mixings` and `mlp_mixings` have `requires_grad=False`; read/write weights and the rest of the network remain trainable.

### Mixer controls

`MHCMixing` gains a validated `identity_blend` in `[0, 1]`:

\[
H=(1-c)I+c\,\operatorname{Sinkhorn}(L/T).
\]

`IsoHCMixing` gains a validated `complement_scale` in `(0, 1]`:

\[
H=P+\alpha U R U^\top.
\]

Defaults remain `identity_blend=1` and `complement_scale=1`, so existing public behavior and checkpoints are preserved. `scaled-isohc` is an experiment label backed by the same IsoHC implementation with `complement_scale<1`; it is not reported as an isometry.

For a noise-free symmetric Birkhoff initialization, the matched single-step complement gain is computed with the standard library:

\[
\alpha(b,T,n)=\frac{\exp(b/T)-1}{\exp(b/T)+n-1}.
\]

The geometry suite verifies this prediction against the actual Sinkhorn output before using it for a scaled-IsoHC control. With nonzero noise, only the measured gain is authoritative.

### Experiment suites

`experiments/hc_causal_controls.py` exposes four suites:

1. `geometry`: no LM training. It scans `diag_bias={2,4,6,8}`, `temperature={0.5,1,2}`, `noise_std={0,0.01}`, and `sinkhorn_iters={5,10,20}` for the requested stream count and transport depth. It records analytic and measured single-step and composite geometry.
2. `p0-smoke`: tiny random-data runs covering identity-HC, IsoHC, one trainable static-Birkhoff configuration, its frozen counterpart, identity-blended Birkhoff, and matched scaled-IsoHC.
3. `p0-train`: the paper-facing shortlist. It runs `diag_bias={2,4,6,8}` with frozen and trainable static Birkhoff, the matched scaled-IsoHC controls, identity-HC, and IsoHC. Temperature variants are promoted from `geometry` only when they materially change the initial spectrum, avoiding an automatic full Cartesian training grid.
4. `p0-depth`: the fixed-parameter depth study. It runs `L={24,48,72,96,128}` and reports `2L` transport steps. The selected widths below were measured with the current model, vocabulary `50257`, eight heads, four streams, context `512`, and tied embeddings; every configuration is within 5% of the 48-layer reference parameter count.

| Layers | Width | Heads | Parameters | Transport steps |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 704 | 8 | 178,516,487 | 48 |
| 48 | 512 | 8 | 177,041,159 | 96 |
| 72 | 416 | 8 | 170,703,431 | 144 |
| 96 | 368 | 8 | 174,765,479 | 192 |
| 128 | 320 | 8 | 173,618,055 | 256 |

The depth suite runs identity-HC, IsoHC, the selected trainable static-Birkhoff configuration, its frozen counterpart, and the matched scaled-IsoHC control. `depth_summary.json` records `log(composite_sv_min)`, `log(composite_sv_max)`, and the corresponding sums of per-step log singular values against both layer count and actual transport count.

Every output directory includes `experiment_variant` and `seed`. The entrypoint refuses to overwrite an existing completed `run_summary.json`; reruns use a new output root or explicit resume workflow rather than deleting evidence.

## Measurements

### Geometry trajectory

`run_summary.json` uses `metric_schema_version=2` and contains:

```text
initial_transport
training_transport_history[]
final_transport
posthoc
```

Each training snapshot is keyed by optimizer step and processed tokens. Per transport and for the final composite it records all exactly `n_streams - 1` singular values through the fixed basis `U`, rather than filtering small values. It also records:

\[
\ell_{\mathrm{mean}\to\perp}=\lVert U^\top H v\rVert_2,
\qquad
\ell_{\perp\to\mathrm{mean}}=\lVert v^\top H U\rVert_2.
\]

Static-Birkhoff diagnostics subtract column sums from a row-shaped ones tensor, fixing the current accidental broadcast. Parameter values, row/column errors, read/write gate values, and frozen/trainable state are retained with the measurements.

### State metrics

The existing norm ratio is retained under the unambiguous name `mean_zero_norm_ratio`:

\[
r_\perp=\frac{\lVert P_\perp X\rVert_F}{\lVert X\rVert_F}.
\]

`mean_zero_energy` becomes the squared ratio:

\[
E_\perp=\frac{\lVert P_\perp X\rVert_F^2}{\lVert X\rVert_F^2}.
\]

The old full-state stream cosine remains for backward comparison, and a new centered stream cosine is reported from `P_perp X`. Historical schema-1 artifacts are never silently compared to schema-2 energy values.

### Dtype and method provenance

The IsoHC documentation and result metadata state the implementation that actually ran: tiny complement matrices are projected in float64 on CPU/CUDA and float32 on MPS, then converted back to the caller dtype. The summary also records `ns_steps`, `use_svd`, `svd_fallback`, caller parameter dtype, AMP dtype, and whether `torch.compile` was enabled. It does not imply that a disabled SVD fallback ran.

Minimal Givens/Householder parameterization is deliberately deferred to P2. Replacing raw-polar IsoHC during P0 would change the optimizer geometry at the same time as the causal controls, defeating the purpose of the first experiment.

### Persistent complement intervention

The model already accepts multiple `state_index` values, so persistent clamping needs no new forward API. For start state `k`, the analyzer applies the intervention at every state from `k` through `2L`:

```python
{
    "state_index": list(range(k, 2 * model.num_layers + 1)),
    "mode": "scale_perp",
    "scale": gamma,
}
```

The default scan is `gamma={0,0.25,0.5,0.75,1}` at states selected by the existing intervention stride. Baseline and intervened logits use the same validation batches. Each point reports validation NLL/PPL, delta NLL, mean token KL from baseline, and top-1 prediction-change rate. Monotonicity is reported from the data; it is not asserted by the code.

The existing single-state removal remains available and is labeled as a weak intervention because later branches may recreate complement state.

## Files and Responsibilities

- `docs/0714_experiment_guidance.md`: verbatim supplied guidance with a non-authoritative status banner.
- `lm/mixing.py`: validated Birkhoff/IsoHC control parameters, accurate projection-dtype documentation, and fixed-basis diagnostics.
- `lm/models.py`: mixer configuration plumbing and unchanged intervention contract.
- `lm/transport_analysis.py`: exact complement spectra, composite trajectory inputs, and mean/complement leakage.
- `lm/diagnostics.py`: schema-2 norm, energy, centered-cosine, and transport snapshot metrics.
- `lm/train.py`: attach optimizer step and token count to periodic diagnostic snapshots.
- `experiments/lm_5090_next_runs.py`: method aliases, config propagation, freeze handling, initial/final reports, and persisted diagnostic history.
- `experiments/hc_causal_controls.py`: only the experiment matrix and safe orchestration; no training implementation.
- `experiments/analyze_lm_mechanisms.py`: paired persistent-intervention evaluation.
- `tests/test_lm_next_phase_contracts.py`: deterministic contracts for configuration, controls, metrics, and interventions.
- `README.md`, `experiments/README.md`, `agent.md`: canonical commands and the static-baseline naming boundary.

## Validation

The implementation is complete only after all of the following run successfully:

```bash
PYTHONDONTWRITEBYTECODE=1 "$HOME/miniconda3/envs/aidemo/bin/python" \
  -m unittest tests.test_lm_next_phase_contracts tests.test_stage1_contracts -q
```

The new tests must prove:

- legacy defaults produce the same mixer behavior;
- configured values reach every attention and MLP mixer;
- frozen mixer parameters have no gradients while the rest of the model remains trainable;
- the noise-free analytic gain matches measured Sinkhorn geometry;
- scaled-IsoHC preserves the mean and applies the requested complement gain;
- depth presets produce exactly `2L` transports and remain within 5% of the measured 48-layer parameter reference;
- exactly `n_streams - 1` singular values and both leakage directions are reported;
- schema-2 norm and energy ratios have the expected square relationship;
- result metadata matches the actual projection, caller, AMP, fallback, and compile settings;
- persistent intervention acts at every requested state and paired metrics are finite;
- old `mhc` checkpoints remain loadable while new output is labeled `static-birkhoff-hc`.

A CPU artifact smoke run then uses `run0`, random tokens, no compile, and a short token budget through `p0-smoke`. Its JSON is parsed with the Python standard library to verify configuration provenance and the initial/training/final fields. This validates orchestration only.

Real FineWeb-Edu, RTX 5090 throughput, multi-seed statistical conclusions, faithful dynamic mHC, BiLip-HC, and 100M–1B scaling remain explicitly unverified until those runs or later-stage implementations are performed.

## P0 Decision Rules

- If frozen and trainable static Birkhoff show the same depth contraction, report initialization/parameterization as the dominant cause; do not attribute it to learned routing.
- If matched scaled-IsoHC reproduces the loss or intervention behavior, attribute the effect to contraction rather than non-negativity.
- If persistent clamp has negligible paired NLL/KL impact, do not spend on 1B scaling; proceed to P1 accessibility tasks and gate scans.
- If effects survive matched controls and multiple seeds, promote only the surviving configurations to P1/P3.
- If faithful dynamic mHC later disagrees with the static proxy, narrow all current conclusions to static Birkhoff-HC.
