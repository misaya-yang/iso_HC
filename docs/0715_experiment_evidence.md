# HC Experiment Evidence Ledger

**Status:** working evidence boundary, updated 2026-07-15. Generated artifacts
under `outputs/` remain local and are not source-controlled.

## Supported

- The current IsoHC transport preserves the fixed mean direction and, within
  measured numerical precision, preserves complement singular values.
- The static Birkhoff/Sinkhorn implementation can contract the complement. It
  remains a static proxy and is not evidence about faithful dynamic mHC.
- The real FineWeb row cache now runs through the canonical loader on MPS.

## Guardrail Only

The matched 125M FineWeb run used one seed and 5M training tokens:

| Method | Validation NLL | Validation PPL |
| --- | ---: | ---: |
| identity-HC | 6.442632 | 628.058 |
| IsoHC | 6.440155 | 626.504 |

The NLL difference is too small and underpowered to establish functional
utility. This run demonstrates training compatibility, not a PPL advantage.

## Not Yet Supported

- The trained LM causally uses complement information.
- IsoHC improves language-model quality or residual scaling.
- Results from the static Birkhoff proxy generalize to official dynamic mHC.
- A shallow synthetic exact-match result is sufficient without NMSE and a
  transport-depth curve.

## Next Decision Gates

1. Do not add more PPL or gate sweeps until a forced-complement task shows
   exact-match, NMSE, and depth-robust causal use.
2. Use the free T4 first for a short memory/throughput smoke. It is a background
   runner, not large-model evidence.
3. Promote to a long LM run only after the mechanism gate passes. A credible
   small-model comparison needs a substantially larger token budget and
   multiple seeds; a scaling claim needs external compute beyond one T4.

## T4 Boundary

The T4 has 16GB memory and should use CUDA FP16 AMP with gradient scaling.
Start with a conservative common batch and a 20M-token calibration run; record
tokens/second, peak memory, non-finite steps, and validation NLL before choosing
any longer budget.
