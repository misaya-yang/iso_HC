# 0525 FE 48L Overnight Summary

Result root: `/root/autodl-tmp/isoHC/results/0525_fe_fair_deep48_p33013_20m`

## Fair Batch Probe

```json
{
  "baseline": {
    "max_fitting_batch": 32,
    "probe_peak_gb": 28.314676761627197
  },
  "identity-hc": {
    "max_fitting_batch": 24,
    "probe_peak_gb": 28.296132564544678
  },
  "unconstrained": {
    "max_fitting_batch": 20,
    "probe_peak_gb": 27.497609615325928
  },
  "mhc": {
    "max_fitting_batch": 20,
    "probe_peak_gb": 27.499486446380615
  },
  "isohc": {
    "max_fitting_batch": 20,
    "probe_peak_gb": 27.498525142669678
  }
}
```

## Runs

| method | success | batch | tokens(M) | tok/s | elapsed(min) | val_loss | val_ppl | E_perp init->final | cosine init->final |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | True | 20 | 20.0 | 48359 | 6.9 | 5.7366 | 310.01 |  ->  |  ->  |
| identity-hc | True | 20 | 20.0 | 34351 | 9.7 | 5.7157 | 303.60 | 0.0222 -> 0.0503 | 0.9993 -> 0.9971 |
| unconstrained | True | 20 | 20.0 | 26333 | 12.7 | 5.6760 | 291.78 | 0.0222 -> 0.0547 | 0.9993 -> 1.0000 |
| mhc | True | 20 | 20.0 | 16846 | 19.8 | 5.7249 | 306.39 | 0.0224 -> 0.0136 | 0.9993 -> 0.9998 |
| isohc | True | 20 | 20.0 | 22498 | 14.8 | 5.7206 | 305.10 | 0.0224 -> 0.0546 | 0.9993 -> 0.9967 |
