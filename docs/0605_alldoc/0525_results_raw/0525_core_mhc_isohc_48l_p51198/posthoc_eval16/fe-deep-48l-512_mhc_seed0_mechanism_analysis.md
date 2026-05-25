# Mechanism Analysis: mhc

- run: `/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l_p51198/fe-deep-48l-512_mhc_seed0`
- checkpoint: `/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l_p51198/fe-deep-48l-512_mhc_seed0/final.pt`
- base val loss/PPL: `5.9007` / `365.30`

## Composite Complement Gain

- final composite sv mean: `0.00043895`
- product of per-step sv mean: `0.000438614`
- final composite sv range: `0.000418797` to `0.000462579`

## Stream Gradient

- slope of log grad ratio: `-0.0211095`

## Replacement Interventions

- replace with identity delta loss: `-0.000523716`
- replace with random fixed-vector orthogonal delta loss: `-0.000831664`

## Complement Removal

- strongest delta loss: `0.000288814` at state `24`
