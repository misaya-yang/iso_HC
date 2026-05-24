# 0524 IsoHC LM 实验报告

## 状态

本轮 `deep-stress-512` 已补跑完成，四个方法都有结果。原始 JSON/log 已拉回到：

- `docs/0605_alldoc/0524_results_raw/0524_deep_stress_512/`
- checkpoint `.pt` 未拉回本地，避免占用存储。

服务器数据盘已准备更大真实数据 cache：

- `/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt`
- `/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt`
- train tokens: `100,002,129`
- heldout tokens: `2,009,986`
- 本地 `/tmp/isohc_fineweb_edu_100m_0524` 已删除。

## Crash 与修复

`isohc` 首次运行挂在 `torch.compile(max-autotune)` 的 CUDA graph capture 阶段。根因是 IsoHC 投影路径触发了 SVD：

```text
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
...
torch.ops.aten._linalg_svd.default
```

修复：

- `isohc/projection.py`: Newton-Schulz 初始化缩放由 spectral norm 改为 Frobenius norm，避免隐式 SVD。
- `lm/models.py` / `experiments/lm_5090_next_runs.py`: LM 训练里的 IsoHC 关闭 SVD fallback，使用纯 finite-step NS projection。
- 本地测试通过：`python3 -m unittest tests.test_lm_next_phase_contracts tests.test_stage1_contracts -v`。
- 远程短 smoke 通过后，重新补跑 `isohc` 完整 20M tokens。

## 训练结果

配置：`L=24, d=512, h=8, T=512, n_streams=4, TinyStories cache, 20M tokens, seed=0`。

| method | batch | val loss | val ppl | train tok/s | elapsed |
|---|---:|---:|---:|---:|---:|
| baseline | 48 | 2.4312 | 11.37 | 153,891 | 134s |
| unconstrained | 12 | 2.1964 | 8.99 | 40,914 | 491s |
| mHC | 12 | 2.2487 | 9.48 | 30,634 | 656s |
| IsoHC | 12 | 2.2602 | 9.58 | 67,539 | 300s |

注意：baseline 因 auto-batch 选到 48，与 HC 系列 batch=12 不完全公平；HC 三者之间更可比。当前不能把这张表写成“PPL 全面胜过 mHC/baseline”。

## 机制指标

| method | mean-zero energy final | stream cosine final | key constraint diagnostics |
|---|---:|---:|---|
| unconstrained | 0.3498 | 0.8300 | fix error mean 0.1488, orth error mean 0.2582 |
| mHC | 0.0668 | 0.9946 | 1-perp sv mean 0.9156, sv max mean 0.9184 |
| IsoHC | 0.1811 | 0.9584 | fix error mean 2.81e-7, orth error mean 5.97e-4, 1-perp sv mean 0.99997 |

这组机制信号和老师建议的主线是一致的：

- mHC 的 `1^\perp` singular values 明显小于 1，出现 complement diffusion/collapse 倾向。
- IsoHC 保住 fixed-vector 约束，`fix_error` 在 `1e-7` 量级，`1^\perp` 奇异值贴近 1。
- unconstrained loss 最好，但 invariant 和 orthogonality 都漂移很大，说明它不是稳定性边界上的安全对照。

## 结论

本轮最重要的结论不是 TinyStories PPL，而是工程和机制闭环：

1. IsoHC 的 compile 崩溃已定位并修复。
2. 在 24-layer deep-stress LM 中，IsoHC 成功完整训练，且比 mHC 更好地保持 `1^\perp` isometry。
3. 当前 TinyStories 20M-token 小实验中，IsoHC loss 略差于 mHC：`2.2602` vs `2.2487`，不能声称性能胜出。
4. 下一步应使用 FineWeb-Edu 100M-token cache 做固定 batch/固定 token 或固定 optimizer-step 的 controlled run。

## 下一步

优先跑：

- `baseline / unconstrained / mHC / IsoHC`
- 固定 `batch=12`，或用 gradient accumulation 做等效 batch，避免 auto-batch 造成对照偏差。
- 数据切到 FineWeb-Edu cache：
  - train: `/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_ctx512.pt`
  - val: `/root/autodl-tmp/isoHC/data/lm_cache/HuggingFaceFW__fineweb-edu__sample-10BT_train_heldout_ctx512.pt`

同时把后续大结果写到 `/root/autodl-tmp/isoHC/results/`，不要再放系统盘。
