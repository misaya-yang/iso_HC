# IsoHC 项目状况总结 — 请给予后续实验规划建议

## 项目概述

我们在做一个名为 **IsoHC (Isotropic Hyper-Connections)** 的深度学习残差传输机制研究。核心想法是：

**问题**：Hyper-Connections (HC) 将标准残差连接扩展为多个 residual streams，并学习跨 stream 的 mixing 矩阵 H。但这个 H 如果不加约束，会在深层指数爆炸（unconstrained HC）；如果用 mHC 的 Birkhoff/双随机约束（H@1=1, 1^T@H=1^T, H≥0），虽然防止了爆炸，但双随机矩阵在均值补空间（1_perp）上是扩散性的，会指数式收缩 mean-zero stream 子空间，导致多 stream 的额外容量在深层被耗散掉。

**我们的方案 IsoHC**：用 fixed-vector isometric 约束替代 Birkhoff 约束：
```
Q^T Q = I,    Q @ 1 = 1
```
即 Q 是正交矩阵且固定 all-ones 向量。这样：
- 均值方向（residual mean）被精确保持 → 标准残差 identity path 不变
- 在 1_perp 补空间上 Q 的作用是等距旋转 → 所有奇异值 = 1，不收缩也不膨胀

**投影方法**：给定无约束矩阵 A，通过 Newton-Schulz polar iteration（K=5步）在正交补空间上做极分解近似，然后重构。Fixed-vector 约束通过构造严格保持（fix_error < 1e-6），正交性是近似的但精度在 ~1e-6 级别。

## 实验证据现状（按说服力排序）

### ✅ 强证据（机制层面）

1. **1024层极限深度检测器**（residual-only，无 attention/MLP）
   - IsoHC: grad=1.0000, energy=1.0000, stream_cosine=0.025 → 完美稳定
   - mHC-lite: grad=0.35, energy=0.0000, cosine=1.0 → 收缩+collapse
   - unconstrained: grad=5.58e13 → 爆炸
   - 所有 36 个 IsoHC 配置 (n=4/8/16, L=32-1024, fp32+bf16) 全部通过

2. **48层 FineWeb-Edu Transformer posthoc 机制分析**（177M params, 20M tokens）
   - mHC: composite 1_perp transport gain 坍塌到 4.39e-4
   - IsoHC: composite 1_perp transport gain 保持在 0.996
   - mHC 的 1_perp 奇异值均值 ~0.922（每层收缩 ~8%）
   - IsoHC 的 1_perp 奇异值均值 ~0.99996（几乎完美等距）

3. **GNN synthetic oversmoothing**（128层深度，512节点 SBM 图）
   - GCN/ResGCN: energy=0.20, variance→0.001, cosine=1.0 → 完全 collapse
   - IsoNode: energy=1.0, variance=1.0, cosine=0.03 → 完美保持

4. **数学正确性**：投影精度 ~1e-6，fix_error < 1e-6，bf16_fp32_mix 精度策略完全恢复 fp32 质量

### ⚠️ 中等证据（下游信号）

5. **Cora node classification**（多 seed, 5 seeds）
   - GCN L16: 31.1% ± 0.2%, L32: 31.2% ± 0.3% (collapse)
   - IsoStream v2 L16: 73.1% ± 1.9%, L32: 66.6% ± 3.4%
   - 从 deep collapse 的 ~30% 恢复到 64-73%，恢复 shallow GCN 性能的 85-95%
   - 但 v1→v2 的架构修复（stream_embed + concat readout）才是关键，不是算子本身

6. **H-type 消融**（固定 v2c 架构，只换 H 类型）
   - L16: identity ≈ iso ≈ none ≈ 74%，所有方法差不多
   - L32: iso (70.7%) 显著优于 identity (66.4%)
   - 结论：主要收益来自 multi-stream + concat 架构本身，isometry 在深层提供额外稳定优势

### ❌ 弱证据 / 主要缺口

7. **LM outcome bridge 非常弱**
   - complement removal 对 val loss 的影响只有 1e-4 到 1e-3 量级
   - 替换 learned IsoHC 为 identity 或 random isometry，loss 变化 < 0.001
   - val loss: mHC=5.9007, IsoHC=5.9013 → 几乎相同，IsoHC 甚至略差
   - **这意味着： preserved complement 还没有被模型真正"用上"**

8. **identity-HC 是一个严肃的 control**
   - identity matrix 本身就是 M_1 的成员（完美等距，但无跨 stream 通信）
   - 当前实验中 identity-HC 和 learned IsoHC 差异极小
   - 如果 learned rotation 始终不比 identity 好，说明架构没有激活 isometric stream communication 的价值

9. **只有单 seed 的 48L LM 结果**，无 seed robustness

10. **没有 depth scaling** 的 LM 实验（只有 48L，没跑 72L/96L）

## 硬件与资源

- GPU: NVIDIA RTX 5090 (32GB)
- 服务器：国内云 GPU（AutoDL），可能无稳定 HuggingFace 访问
- 数据：FineWeb-Edu token cache 已预处理好（100M tokens），存在数据盘
- 当前模型规模：177M params (48L, d=512, 8 heads, 4 streams)
- 一次 48L/20M token 的 3-method 比较约需 45-80 分钟

## 当前论文定位

我们已经写了一个 arxiv 草稿，定位为 **mechanism note**，不做大规模 LM PPL 优越性声明。核心论点是：

> "Birkhoff stability is not isometric stability. mHC preserves the residual mean but dissipates the residual complement. IsoHC is the fixed-vector isometric repair for that missing geometry."

论文的 7 个 Limitations 已经写明：
1. 无大规模 LM 优越性声明
2. intervention effect 弱
3. identity-HC 是严肃 control
4. NS 是近似投影
5. 当前是 mechanism note，不是 full empirical method paper

## 已经规划的后续实验（在我们的 followup plan 中）

1. **Seed robustness**: 48L/20M 跑 3 seeds
2. **Depth scaling**: L ∈ {24, 48, 72}，看 mHC contraction 是否随 depth 加速
3. **100M token bridge**: 仅当 intervention signal 变强后才做
4. **Stop conditions**: 如果 IsoHC→identity delta_loss 持续接近零，就接受更窄的 claim

## 我的核心问题

请基于以上全部信息，回答以下问题：

### 1. 这个项目值不值得继续做？
- 当前的 evidence 已经足以发一篇 mechanism note（我们已经写了草稿）。但要升级为有影响力的 method paper，缺什么？
- identity-HC control 的问题有多严重？如果 learned IsoHC 始终不比 identity 好，这个项目是否就 dead end 了？
- 从审稿人角度看，这个工作的最大风险和最大亮点分别是什么？

### 2. 后续实验如何优化？
- 我们目前的计划是 seed robustness → depth scaling → 100M token bridge。这个优先级对吗？
- 有没有什么**低成本高信息量**的实验我们还没想到？比如：
  - synthetic routing / associative recall task（强制跨 stream 通信）
  - token-level intervention（per-token KL、tail-token loss）而非平均 loss
  - deep-thin stress models（L=96/128，缩小 d_model）
  - 不同 stream 数量 (n=8/16) 的 LM 实验
- 如果 intervention signal 始终弱，有什么方法可以"强制"模型用上 complement？
  - 比如：设计一个任务明确需要多条独立信息流
  - 或者：在 training 过程中对 mean stream 加 noise，迫使模型使用 complement

### 3. 论文策略
- 是应该继续走 mechanism note 路线，还是值得投入更多计算去争一个 outcome paper？
- GNN 方向的 Cora 结果能否成为论文的第二支柱？还是需要更多数据集（Citeseer/PubMed）？
- 如何 best frame "identity-HC 也很强" 这个事实？是 weakness 还是可以变成 insight？

### 4. 技术优化方向
- 当前 NS 投影有 ~35% 的训练 overhead。有什么方法可以降？（Triton kernel？更少的 NS steps？Learned retraction？）
- bf16_fp32_mix 是当前的精度策略。有没有可能做到 bf16-friendly 的 fixed-vector projection？
- stream 数量 n=4 是当前默认值。这个值是否太低以至于 complement 空间太小（只有 3 维），不足以承载有用信息？

### 5. 最大的 unknown unknowns
- 我们可能遗漏了什么关键的实验或分析？
- 有没有什么 theoretical 结果可以 strengthen 或 weaken 我们的 claim？
- 相关工作中有没有我们没注意到的直接竞争对手？（我们目前知道的是：Hyper-Connections [Zhu et al. 2024], mHC [Xie et al. 2025], Spectral-Sphere HC [Liu et al. 2026]）

请给出具体的、可执行的建议，而非泛泛的方向。我们的 GPU 资源有限（单卡 5090），需要在有限预算内做最大化信息增益的实验。
