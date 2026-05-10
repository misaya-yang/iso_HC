# 0510 IsoHC 实验报告

## 结论摘要

本轮实验支持 IsoHC 的核心数学与工程集成可行，但也明确暴露了 mixed precision 下的数值漂移问题。

- Projection sanity 在 RTX 5090 / CUDA 上通过，`n=4/8/16, K=5` 的 orth/fix/energy 指标全部达标。
- Residual-only 的 energy 和 grad 在 fp64/fp32/bf16-fp32-mix 下都保持得很好，说明等距传播目标基本成立。
- `bf16_all` 会让 mean preservation error 明显变大，L=256/512 时达到 `1e-1` 到 `1e0` 量级。这是下一版方法或论文必须正面处理的数值点。
- Tiny Transformer depth scaling smoke 中，baseline 和 IsoHC 在 16/32/48/64 层、300 steps 下全部稳定通过。
- IsoHC 在所有 smoke 中 projection error 稳定在 `~1e-6` 以下，没有 NaN、Inf、loss explosion 或 grad explosion。
- 新增 h512/context512 真实文本 followup 显示：32 层时 IsoHC val loss `1.7194`，baseline val loss `1.9158`；但这是单 seed、非严格 token-budget 结论，只能作为继续放大的信号。

当前最合理的论文定位是：IsoHC 是一个可训练、近等距、均值方向受控的 residual mixing 机制；它的理论性质成立，但低精度实现需要 fp32 mixing 或补偿策略。

## 服务器与环境

- Server: `root@connect.westd.seetacloud.com -p 19596`
- GPU: NVIDIA GeForce RTX 5090
- PyTorch: `2.8.0+cu128`
- Raw results copied to:
  - `docs/0605_alldoc/0510_results_raw/stage1_cuda_projection_leftfix/`
  - `docs/0605_alldoc/0510_results_raw/stage1_cuda_smoke_manual/`
  - `docs/0605_alldoc/0510_results_raw/stage2_precision/`
  - `docs/0605_alldoc/0510_results_raw/stage2_depth_scaling/`
  - `docs/0605_alldoc/0510_results_raw/stage2_real_text/`
  - `docs/0605_alldoc/0510_results_raw/stage2_real_text_grid_h512/`
  - `docs/0605_alldoc/0510_results_raw/stage2_real_text_grid_h512_followup/`

## Experiment A: Precision / Depth Suite

配置：

- streams: `4, 8`
- depths: `32, 64, 128, 256, 512`
- feature_dim: `512`
- trials: `16`
- variants: `fp64`, `fp32`, `bf16_all`, `bf16_fp32_mix`
- TF32 disabled for fp32 diagnostics

关键结果如下，只列 L=256/512。

| variant | n | L | energy mean | grad mean | mean max | orth max |
|---|---:|---:|---:|---:|---:|---:|
| fp64 | 4 | 256 | 0.99999992 | 0.99999991 | 7.73e-07 | 1.04e-06 |
| fp64 | 4 | 512 | 0.99999983 | 0.99999983 | 8.24e-07 | 1.03e-06 |
| fp64 | 8 | 256 | 1.00000001 | 1.00000000 | 6.71e-07 | 3.24e-07 |
| fp64 | 8 | 512 | 1.00000000 | 1.00000002 | 6.65e-07 | 3.12e-07 |
| fp32 | 4 | 256 | 0.99999991 | 0.99999991 | 1.15e-05 | 8.88e-07 |
| fp32 | 4 | 512 | 0.99999988 | 0.99999987 | 1.68e-05 | 9.34e-07 |
| fp32 | 8 | 256 | 1.00000005 | 1.00000002 | 8.12e-06 | 2.98e-07 |
| fp32 | 8 | 512 | 1.00000002 | 1.00000005 | 1.18e-05 | 3.20e-07 |
| bf16_all | 4 | 256 | 1.00292897 | 1.00363389 | 6.22e-01 | 9.91e-07 |
| bf16_all | 4 | 512 | 0.99917658 | 0.99913976 | 6.95e-01 | 1.05e-06 |
| bf16_all | 8 | 256 | 0.99907121 | 0.99886410 | 3.50e-01 | 2.88e-07 |
| bf16_all | 8 | 512 | 0.99696363 | 0.99658824 | 5.53e-01 | 2.94e-07 |
| bf16_fp32_mix | 4 | 256 | 0.99999989 | 0.99999986 | 1.05e-05 | 9.88e-07 |
| bf16_fp32_mix | 4 | 512 | 0.99999985 | 0.99999984 | 1.55e-05 | 9.74e-07 |
| bf16_fp32_mix | 8 | 256 | 1.00000001 | 0.99999998 | 7.87e-06 | 3.02e-07 |
| bf16_fp32_mix | 8 | 512 | 1.00000003 | 1.00000004 | 1.12e-05 | 3.00e-07 |

Interpretation:

- Energy 和 grad preservation 非常强，几乎不随深度恶化。
- mean preservation 的理论性质在 fp64/fp32 下基本成立，但 fp32 在 n=4、深层时会贴近 `1e-5` 阈值。
- `bf16_all` 的 mean drift 很大，说明不能把 H 和 state 全部放到 bf16 后还要求严格 mean preservation。
- `bf16_fp32_mix` 几乎恢复 fp32 行为，是实际工程路径的优先候选。

## Experiment B: Depth Scaling Smoke

配置：

- synthetic LM data
- hidden_dim: `256`
- context_length: `256`
- heads: `4`
- streams: `4`
- precision: `bf16`
- steps: `300`
- batch_size: `8`
- grad_accum_steps: `4`
- layers: `16, 32, 48, 64`

| model | layers | final loss | max loss | max grad norm | time | max orth | max fix |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 8.3219 | 8.3710 | 0.7466 | 74.7s | n/a | n/a |
| baseline | 32 | 8.3230 | 8.3603 | 0.7813 | 106.8s | n/a | n/a |
| baseline | 48 | 8.3218 | 8.3690 | 0.7643 | 126.6s | n/a | n/a |
| baseline | 64 | 8.3234 | 8.3617 | 0.8439 | 152.7s | n/a | n/a |
| IsoHC | 16 | 8.3230 | 8.3628 | 0.7558 | 113.3s | 6.60e-07 | 4.00e-07 |
| IsoHC | 32 | 8.3219 | 8.3824 | 0.7737 | 158.5s | 6.62e-07 | 3.77e-07 |
| IsoHC | 48 | 8.3233 | 8.3728 | 0.7772 | 227.9s | 6.62e-07 | 4.04e-07 |
| IsoHC | 64 | 8.3223 | 8.3694 | 0.8384 | 303.6s | 6.70e-07 | 4.34e-07 |

Interpretation:

- 16 到 64 层的 synthetic smoke 全部稳定。
- IsoHC 没有引入明显训练不稳定性。
- IsoHC 的投影误差在训练中没有漂移。
- 这个实验不能证明 IsoHC 性能优于 baseline，因为 synthetic random-token LM 的 loss 没有真实语义学习目标；它主要证明深度扩展时工程集成稳定。

## Experiment C: Real Text Byte-Level LM

配置：

- Dataset: TinyShakespeare plain text
- Task: byte-level next-token prediction
- vocab_size: `256`
- hidden_dim: `256`
- context_length: `256`
- batch_size: `32`
- precision: `bf16`
- steps: `1000`
- layers: `16, 32`

| model | layers | final train loss | val loss | max grad norm | time | max orth | max fix |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 1.3796 | 1.6104 | 10.1626 | 67.6s | n/a | n/a |
| baseline | 32 | 1.4144 | 1.5903 | 10.6153 | 98.5s | n/a | n/a |
| IsoHC | 16 | 1.3833 | 1.6122 | 10.1649 | 96.7s | 1.87e-04 | 3.58e-07 |
| IsoHC | 32 | 1.4154 | 1.5897 | 10.6172 | 133.9s | 8.97e-05 | 4.04e-07 |

Interpretation:

- Real text 上 baseline 和 IsoHC 都稳定学习，loss 从约 `5.5` 降到 `1.4` 左右。
- 16 层时 IsoHC val loss 与 baseline 基本持平，略差 `+0.0018`。
- 32 层时 IsoHC val loss 与 baseline 基本持平，略好 `-0.0007`。
- 这不是显著性能优势，但它是一个重要信号：IsoHC 在真实文本训练里没有破坏优化，也没有出现投影漂移。
- IsoHC 训练中的 orth error 从 synthetic smoke 的 `~1e-6` 上升到 `1e-4` 量级，但仍远小于 `5e-3` 阈值；fix error 仍保持 `~4e-7`。

## Experiment D: h512 / context512 Real Text Followup

配置：

- Dataset: TinyShakespeare plain text
- Task: byte-level next-token prediction
- hidden_dim: `512`
- context_length: `512`
- heads: `8`
- precision: `bf16`
- steps: `2000`
- streams: `8` for IsoHC

| model | layers | batch | params | final train loss | val loss | max grad norm | mean step ms | max orth | max fix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16 | 32 | 67.52M | 0.1054 | 3.1542 | 15.5140 | 172.3 | n/a | n/a |
| IsoHC | 16 | 32 | 67.53M | 0.1135 | 3.1324 | 15.5207 | 221.0 | 6.85e-04 | 4.04e-07 |
| baseline | 32 | 16 | 134.64M | 0.6384 | 1.9158 | 16.4590 | 180.0 | n/a | n/a |
| IsoHC | 32 | 16 | 134.66M | 0.8627 | 1.7194 | 16.4688 | 783.1 | 4.48e-04 | 4.21e-07 |

Interpretation:

- 16 层 h512 下 baseline 与 IsoHC 都明显过拟合，val loss 都在 `3.1` 附近；IsoHC 略好，但不能作为主要证据。
- 32 层 h512 下 IsoHC val loss 比 baseline 低 `0.1964`。这是目前最积极的真实文本信号，但仍只有单 seed，而且 IsoHC run 后半段与 stability detector 并行，step time 被显著拖慢。
- IsoHC 的 energy ratio 在训练日志中持续贴近 `1.0000`，fix error 仍稳定在 `~4e-7`；orth error 在 h512 长训练下升至 `~4e-4` 到 `~7e-4`，仍低于当前 smoke 阈值。
- 这组实验可以写成“放大后未崩，并出现有利信号”，但不应直接声称 perplexity 胜出；需要多 seed、固定 token budget、关闭并行干扰后重跑。

## Experiment E: Stability Boundary Detectors

本实验已经在服务器完成，目标是替代单纯 PPL 曲线，直接测论文主张需要的稳定性边界：

- gradient norm vs depth
- mean-zero subspace energy retention
- stream cosine similarity / collapse
- train-collapse boundary by method and precision

配置：

- methods: `isohc`, `mhc-lite`, `unconstrained`, `gcn-diffusion`
- streams: `4, 8, 16`
- depths: `32, 64, 128, 256, 512, 1024`
- dtypes: `fp32`, `bf16_fp32_mix`
- feature_dim: `1024`
- trials: `16`

关键 L=1024 结果：

| dtype | method | streams | L | grad ratio | mean-zero energy | stream abs cosine | sigma min | sigma max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | IsoHC | 4 | 1024 | 1.0000 | 1.0000 | 0.0228 | 1.0000 | 1.0000 |
| fp32 | IsoHC | 8 | 1024 | 1.0000 | 1.0000 | 0.0250 | 1.0000 | 1.0000 |
| fp32 | IsoHC | 16 | 1024 | 1.0000 | 1.0000 | 0.0249 | 1.0000 | 1.0000 |
| fp32 | mHC-lite | 4 | 1024 | 0.4969 | 0.0000 | 1.0000 | 0.6999 | 1.0005 |
| fp32 | mHC-lite | 8 | 1024 | 0.3536 | 0.0000 | 1.0000 | 0.5724 | 1.0000 |
| fp32 | mHC-lite | 16 | 1024 | 0.2510 | 0.0000 | 1.0000 | 0.4085 | 1.0000 |
| fp32 | gcn-diffusion | 4 | 1024 | 0.5021 | 0.0000 | 1.0000 | 0.8000 | 1.0000 |
| fp32 | gcn-diffusion | 8 | 1024 | 0.3533 | 0.0000 | 1.0000 | 0.8000 | 1.0000 |
| fp32 | gcn-diffusion | 16 | 1024 | 0.2488 | 0.0000 | 1.0000 | 0.8000 | 1.0000 |
| fp32 | unconstrained | 4 | 1024 | 3.39e5 | 3.35e5 | 0.9818 | 0.5305 | 1.5002 |
| fp32 | unconstrained | 8 | 1024 | 5.58e13 | 5.61e13 | 0.9877 | 0.4270 | 1.6155 |
| fp32 | unconstrained | 16 | 1024 | inf | inf | 0.0000 | 0.2879 | 1.7899 |
| bf16_fp32_mix | IsoHC | 4 | 1024 | 1.0000 | 1.0000 | 0.0258 | 1.0000 | 1.0000 |
| bf16_fp32_mix | IsoHC | 8 | 1024 | 1.0000 | 1.0000 | 0.0256 | 1.0000 | 1.0000 |
| bf16_fp32_mix | IsoHC | 16 | 1024 | 1.0000 | 1.0000 | 0.0257 | 1.0000 | 1.0000 |
| bf16_fp32_mix | mHC-lite | 4 | 1024 | 0.5012 | 0.0009 | 1.0000 | 0.7079 | 1.0011 |
| bf16_fp32_mix | mHC-lite | 8 | 1024 | 0.3545 | 0.0007 | 1.0000 | 0.5631 | 1.0000 |
| bf16_fp32_mix | mHC-lite | 16 | 1024 | 0.2492 | 0.0006 | 1.0000 | 0.4071 | 1.0000 |
| bf16_fp32_mix | gcn-diffusion | 4 | 1024 | 0.5014 | 0.0009 | 1.0000 | 0.8000 | 1.0000 |
| bf16_fp32_mix | gcn-diffusion | 8 | 1024 | 0.3523 | 0.0007 | 1.0000 | 0.8000 | 1.0000 |
| bf16_fp32_mix | gcn-diffusion | 16 | 1024 | 0.2506 | 0.0006 | 1.0000 | 0.8000 | 1.0000 |

Interpretation:

- 这正好验证了新的检测器方向：IsoHC 在 `fp32` 与 `bf16_fp32_mix` 下，到 L=1024 仍保持梯度、mean-zero 能量和奇异值边界。
- mHC-lite 很快把 mean-zero 子空间能量压到 0，stream cosine 接近 1，表现为 collapse；stream 越多，梯度比例越低，n=16 时约 `0.25`。
- gcn-diffusion 呈现相同 collapse 形态，mean-zero energy 接近 0，cosine 为 1。
- unconstrained HC 是另一侧失败边界：不是塌缩，而是随深度指数式爆炸，n=16/L=1024 直接出现 `inf`。
- 这组实验应该成为论文主图之一：IsoHC 不是在 PPL 曲线上争小数点，而是在深度传播的稳定性边界上把“收缩 collapse”和“爆炸 instability”同时排除。

## GPU 利用率说明

观察到 GPU 未完全跑满，这是合理的：

- 模型仍偏小，尤其 hidden_dim=256、seq_len=256。
- micro-batch 只有 8，且用了 grad accumulation，GPU 每次实际处理的小 batch 较小。
- IsoHC smoke 每 20 step 会做 projection diagnostics，也会增加 CPU/GPU 同步。

后续如果目标是吞吐而非可比 smoke，可以改：

- batch_size: `16` 或 `32`
- grad_accum_steps: 相应降低或保持，看是要固定 optimizer step 还是固定 token budget
- context_length: `512`
- hidden_dim: `384/512`

但需要注意，大 batch 会改变每 step token 数；如果要和当前 300-step smoke 直接比较，应改为固定 total tokens，而不是固定 steps。

## 当前论文判断

正面证据：

- IsoHC 的投影数学正确。
- 深层 residual-only 的 energy/grad preservation 很强。
- Transformer 集成在 16/32/48/64 层下稳定。
- Projection error 训练中稳定，未观察到漂移。

主要风险：

- `bf16_all` 下 mean preservation 不成立，且漂移很大。
- 当前 smoke 是 synthetic LM，不能支持性能优势结论。
- 还缺真实数据、多 seed、token-budget 对齐、ablation 和吞吐成本分析。

建议论文叙事：

1. 先把 IsoHC 定位为“等距 residual mixing / stability mechanism”，不要一开始承诺 perplexity 优势。
2. 把 mixed precision 漂移作为一个重要工程发现：IsoHC 需要 fp32 projection 和 fp32 mixing path，或者需要专门的 mean-preserving correction。
3. 下一步必须跑真实数据 LM，才能判断是否有性能型 paper 的牙齿。

## 下一步实验

优先级最高：

1. 真实数据小 LM：
   - TinyStories 或 WikiText
   - baseline vs IsoHC
   - 16/32/64 layers
   - 3 seeds
   - 固定 total tokens，不只固定 steps

2. IsoHC precision ablation：
   - bf16 all
   - bf16 activations + fp32 mixing
   - fp32 projection + bf16 rest

3. Cost analysis：
- step time
- memory
- tokens/sec
- projection diagnostics overhead on/off

当前 GPU 利用率仍不高。batch_size=32 时显存约 4GB，利用率约 30%。如果要做正式吞吐实验，应加大模型或 token budget，而不是只加 batch：

- hidden_dim: `384` 或 `512`
- context_length: `512`
- batch_size: `32-64`
- 固定 total tokens 进行比较

4. Stronger baselines:
   - standard residual
   - unconstrained HC
   - mHC-lite

## 文件索引

- Precision suite raw JSON: `0510_results_raw/stage2_precision/precision_depth_suite.json`
- Precision suite Markdown: `0510_results_raw/stage2_precision/precision_depth_suite.md`
- Depth scaling logs: `0510_results_raw/stage2_depth_scaling/*.log`
- Depth scaling status: `0510_results_raw/stage2_depth_scaling/run_status.log`
- Real text JSON/Markdown: `0510_results_raw/stage2_real_text/real_text_*.json`
- Real text logs: `0510_results_raw/stage2_real_text/*.log`
- h512 real text followup: `0510_results_raw/stage2_real_text_grid_h512*/`
- Stability detector: `0510_results_raw/stage2_stability_detectors_core/stability_detectors.json`
- Stability detector Markdown: `0510_results_raw/stage2_stability_detectors_core/stability_detectors.md`
- Stage 1 projection: `0510_results_raw/stage1_cuda_projection_leftfix/projection_sanity.md`
- Stage 1 smoke: `0510_results_raw/stage1_cuda_smoke_manual/`
