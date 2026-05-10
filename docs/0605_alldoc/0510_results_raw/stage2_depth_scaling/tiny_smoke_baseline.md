# Stage 1: Tiny Transformer Smoke Test (baseline)

**Timestamp:** 2026-05-10T22:57:05.805071

**Overall:** PASS

## Configuration

- model_type: baseline
- num_layers: 64
- hidden_dim: 256
- num_heads: 4
- n_streams: 4
- context_length: 256
- vocab_size: 4096
- steps: 300
- eval_every: 100
- batch_size: 8
- grad_accum_steps: 4
- precision: bf16
- grad_clip: 1.0
- learning_rate: 0.0003
- weight_decay: 0.1
- warmup_steps: 50
- ns_steps: 5
- dropout: 0.0
- mlp_ratio: 4

## Metrics

| step | loss | grad_norm | lr | max_orth | max_fix | time_ms |
|-----:|-----:|----------:|---:|---------:|--------:|--------:|
| 0 | 8.3617 | 0.8439 | 0.00e+00 | n/a | n/a | 340.7 |
| 20 | 8.3572 | 0.2217 | 1.20e-04 | n/a | n/a | 183.5 |
| 40 | 8.3316 | 0.1865 | 2.40e-04 | n/a | n/a | 180.1 |
| 60 | 8.3270 | 0.1826 | 2.99e-04 | n/a | n/a | 180.2 |
| 80 | 8.3344 | 0.1849 | 2.89e-04 | n/a | n/a | 181.5 |
| 100 | 8.3260 | 0.1778 | 2.71e-04 | n/a | n/a | 177.3 |
| 120 | 8.3260 | 0.1750 | 2.46e-04 | n/a | n/a | 178.2 |
| 140 | 8.3263 | 0.1776 | 2.14e-04 | n/a | n/a | 182.1 |
| 160 | 8.3248 | 0.1767 | 1.78e-04 | n/a | n/a | 184.6 |
| 180 | 8.3207 | 0.1749 | 1.41e-04 | n/a | n/a | 180.1 |
| 200 | 8.3255 | 0.1710 | 1.04e-04 | n/a | n/a | 180.3 |
| 220 | 8.3198 | 0.1747 | 6.96e-05 | n/a | n/a | 178.6 |
| 240 | 8.3199 | 0.1725 | 4.07e-05 | n/a | n/a | 178.5 |
| 260 | 8.3193 | 0.1716 | 1.86e-05 | n/a | n/a | 177.8 |
| 280 | 8.3180 | 0.1720 | 4.71e-06 | n/a | n/a | 176.3 |
| 299 | 8.3234 | 0.1745 | 1.18e-08 | n/a | n/a | 177.8 |
