# Mechanism Analysis: isohc

- run: `/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l_p51198/fe-deep-48l-512_isohc_seed0`
- checkpoint: `/root/autodl-tmp/isoHC/results/0525_core_mhc_isohc_48l_p51198/fe-deep-48l-512_isohc_seed0/final.pt`
- base val loss/PPL: `5.9013` / `365.50`

## Composite Complement Gain

- final composite sv mean: `0.996349`
- product of per-step sv mean: `0.996343`
- final composite sv range: `0.99057` to `1.00227`

## Stream Gradient

- slope of log grad ratio: `-0.0213729`

## Replacement Interventions

- replace with identity delta loss: `0.000712991`
- replace with random fixed-vector orthogonal delta loss: `0.000559598`

## Complement Removal

- strongest delta loss: `0.000516444` at state `24`
