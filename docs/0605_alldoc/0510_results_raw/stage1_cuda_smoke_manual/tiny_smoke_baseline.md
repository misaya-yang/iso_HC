# Stage 1: Tiny Transformer Smoke Test (baseline)

**Timestamp:** 2026-05-10T21:53:05.688913

**Overall:** PASS

## Configuration

- model_type: baseline
- num_layers: 16
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
| 0 | 8.3744 | 0.6775 | 0.00e+00 | n/a | n/a | 180.7 |
| 20 | 8.3464 | 0.2885 | 1.20e-04 | n/a | n/a | 81.4 |
| 40 | 8.3357 | 0.2015 | 2.40e-04 | n/a | n/a | 54.3 |
| 60 | 8.3275 | 0.1859 | 2.99e-04 | n/a | n/a | 63.9 |
| 80 | 8.3316 | 0.1876 | 2.89e-04 | n/a | n/a | 85.8 |
| 100 | 8.3272 | 0.1802 | 2.71e-04 | n/a | n/a | 81.5 |
| 120 | 8.3277 | 0.1768 | 2.46e-04 | n/a | n/a | 80.4 |
| 140 | 8.3226 | 0.1789 | 2.14e-04 | n/a | n/a | 88.9 |
| 160 | 8.3237 | 0.1768 | 1.78e-04 | n/a | n/a | 82.5 |
| 180 | 8.3223 | 0.1762 | 1.41e-04 | n/a | n/a | 81.8 |
| 200 | 8.3263 | 0.1719 | 1.04e-04 | n/a | n/a | 88.7 |
| 220 | 8.3203 | 0.1752 | 6.96e-05 | n/a | n/a | 81.4 |
| 240 | 8.3215 | 0.1737 | 4.07e-05 | n/a | n/a | 82.0 |
| 260 | 8.3189 | 0.1718 | 1.86e-05 | n/a | n/a | 82.9 |
| 280 | 8.3175 | 0.1732 | 4.71e-06 | n/a | n/a | 81.6 |
| 299 | 8.3210 | 0.1748 | 1.18e-08 | n/a | n/a | 80.8 |
