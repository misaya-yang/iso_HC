# 0525 FE 48L Overnight Report

## Status

Run completed successfully on `root@connect.westd.seetacloud.com:33013`.

Raw results were pulled to:

```text
/Users/yang/projects/isoHC/docs/0605_alldoc/0525_results_raw/0525_fe_fair_deep48_p33013_20m
```

## Setup

- Data: FineWeb-Edu token cache, `sample-10BT`, GPT-2 tokenizer, context length 512.
- Model: `fe-deep-48l-512`, 48 layers, `d_model=512`, 8 heads, 4 streams, about 177M params.
- Budget: 20M tokens per method, 1 seed.
- Methods: `baseline`, `identity-hc`, `unconstrained`, `mhc`, `isohc`.
- Fairness: all methods used common `batch=20`, selected by fair batch probe.

Fair batch probe:

| method | max fitting batch | probe peak GB |
|---|---:|---:|
| baseline | 32 | 28.31 |
| identity-hc | 24 | 28.30 |
| unconstrained | 20 | 27.50 |
| mHC | 20 | 27.50 |
| IsoHC | 20 | 27.50 |

## Results

| method | val loss | val PPL | tok/s | elapsed min | `E_perp` init -> final | stream cosine init -> final |
|---|---:|---:|---:|---:|---|---|
| baseline | 5.7366 | 310.01 | 48,359 | 6.9 | n/a | n/a |
| identity-hc | 5.7157 | 303.60 | 34,351 | 9.7 | 0.0222 -> 0.0503 | 0.9993 -> 0.9971 |
| unconstrained | 5.6760 | 291.78 | 26,333 | 12.7 | 0.0222 -> 0.0547 | 0.9993 -> 1.0000 |
| mHC | 5.7249 | 306.39 | 16,846 | 19.8 | 0.0224 -> 0.0136 | 0.9993 -> 0.9998 |
| IsoHC | 5.7206 | 305.10 | 22,498 | 14.8 | 0.0224 -> 0.0546 | 0.9993 -> 0.9967 |

## Mechanism

| method | fixed-vector error | orth error | `1_perp` singular values |
|---|---:|---:|---|
| identity-hc | 0.0 | 0.0 | exact identity |
| unconstrained | 0.1050 | 0.1694 | unconstrained |
| mHC | row err 0.0046 / col err 2.30e-7 | n/a | mean 0.9222, min 0.9208, max 0.9237 |
| IsoHC | 2.79e-7 | 5.57e-4 | mean 0.99996, min 0.99982, max 1.00012 |

## Takeaways

This run strongly supports the mechanism claim:

- mHC preserves stochastic constraints but contracts the mean-zero stream subspace: `E_perp` falls from `0.0224` to `0.0136`, and `1_perp` singular values average about `0.922`.
- IsoHC preserves the fixed vector and keeps `1_perp` singular values essentially at 1.0, while maintaining much higher final `E_perp`: `0.0546`.
- IsoHC slightly beats mHC on FE 48L validation loss in this run: `5.7206` vs `5.7249`, though the margin is small.
- Unconstrained has the best loss, but violates the intended geometry: fixed-vector error `0.1050`, orthogonality error `0.1694`, and final stream cosine collapses to `0.99997`.

Safe claim for the paper: IsoHC is not yet a clear PPL winner over all alternatives, but it does exactly what the theory predicts: it keeps the invariant while preventing Birkhoff-style contraction in the invariant-complement subspace.
