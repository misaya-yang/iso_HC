<!-- doc-status: historical -->
> **历史记录：不再作为当前研究判断、投稿主张或执行计划。**
> 本文原文与数值保留；当前解释见[证据台账](../research/evidence.md)，理论边界见[理论基础](../research/theory.md)，研究目标与下一实验以[当前主线](../research/README.md)和[实验计划](../research/roadmap.md)为准。本文旧启动、删除、工具调用与投稿指令不再生效。

<!-- doc-history-body-begins -->
# 0525 Posthoc Eval16 Mechanism Report

## Context

This is a higher-confidence posthoc rerun on the completed 48-layer FineWeb-Edu
checkpointed run:

```text
/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l_p51198
```

The analysis used:

```text
batch_size = 1
eval_batches = 16
intervention_stride = 4
methods = mHC, IsoHC
```

It is intentionally a validation/posthoc analysis, not a new training run. The
small batch avoids OOM during gradient and intervention probes.

Raw files were copied to:

```text
docs/0605_alldoc/0525_results_raw/0525_core_mhc_isohc_48l_p51198/posthoc_eval16/
```

## Key Results

| Method | Base val loss | Base PPL | Final composite sv mean on 1_perp | Grad slope | Best complement-removal delta | Replace with identity delta | Replace with random Iso delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| mHC | 5.900727 | 365.30 | 0.000439 | -0.021109 | +0.000289 | -0.000524 | -0.000832 |
| IsoHC | 5.901274 | 365.50 | 0.996349 | -0.021373 | +0.000516 | +0.000713 | +0.000560 |

## Interpretation

The mechanism result is very strong and stable under a larger validation probe:

```text
mHC:   composite 1_perp transport gain collapses to about 4.39e-4
IsoHC: composite 1_perp transport gain stays near 0.996
```

So the central geometry claim is clean:

```text
Birkhoff/mHC preserves the residual mean but dissipates the residual complement.
IsoHC preserves the residual mean and keeps complement transport approximately
isometric.
```

The outcome bridge is still weak. Complement removal changes loss only at the
1e-4 to 1e-3 scale, and replacing learned IsoHC with identity/random fixed-vector
Iso changes loss by less than 0.001 on this checkpoint. This means the current
48L/20M-token run proves the contraction pathology, but does not yet prove that
the preserved complement signal is strongly used by the LM at this budget.

## Decision

Do not frame the current result as "IsoHC beats mHC on LM PPL." The safer and
stronger framing is:

```text
IsoHC is a fixed-vector isometric repair for a measurable depth-wise contraction
pathology in Birkhoff/mHC stream transport.
```

The next useful experiments should not be more variants. They should test whether
the preserved complement becomes functionally important:

1. Run 48L/20M for extra seeds to confirm the mechanism is stable.
2. Run depth scaling, especially 72L, because the claim is depth contraction.
3. Only if the intervention signal grows, run a longer mHC-vs-IsoHC token bridge.

