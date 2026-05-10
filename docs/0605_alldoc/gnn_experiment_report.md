# IsoHC GNN 方向实验报告

**日期**: 2026-05-11
**硬件**: NVIDIA RTX 5090 (32GB)
**环境**: PyTorch 2.8.0+cu128, Python 3.12.3

---

## 1. 核心结论

本报告验证 fixed-vector isometric operator 在图神经网络（GNN）中的应用价值。核心发现：

> **IsoStreamGCN v2 在 16-32 层深度下将 Cora node classification accuracy 从 deep GCN collapse 的 ~30% 恢复到 64-69%，同时保持表示方差不 collapse。**

关键前提是：v1 架构因 stream symmetry 和 mean readout 导致算子未真正发挥作用；v2 修复后 downstream signal 立刻出现。

---

## 2. 背景与动机

### 2.1 GNN Oversmoothing 问题

标准 GCN 的传播：

```
X_{l+1} = σ(S X_l W_l)
```

其中 S = D^{-1/2} A D^{-1/2} 是 symmetric normalized adjacency。深层 GCN 存在明确的 oversmoothing/collapse 现象：节点表示趋于相同，方差归零，分类准确率崩塌。

### 2.2 Iso 算子的理论定位

fixed-vector isometric operator Q 满足：

```
Q^T Q = I,   Qv = v   (v = sqrt(d_tilde))
```

在 GNN 中的角色不是替代 graph diffusion，而是作为**稳定通道**（anti-collapse path），与 GCN diffusion 协同工作。

### 2.3 v1 → v2 的架构演进

| 问题 | v1 | v2 修复 |
|------|-----|---------|
| Stream 初始状态 | 完全相同 (repeat) | stream_embed (确定性差异) |
| Dropout 位置 | 打在 residual state | 仅 message branch |
| Readout | stream mean (抵消 H 作用) | concat (保留所有信息) |
| 结果 L16 | 25.9% | **69.1%** |
| 结果 L32 | 23.6% | **63.9%** |

---

## 3. Synthetic Oversmoothing 机制验证

### 3.1 实验设置

- Graph: 512-node SBM (2 classes)
- Depths: 1, 2, 4, 8, 16, 32, 64, 128
- Methods: GCN, Residual, IsoNode
- Metrics: energy ratio, centered variance, v-centered variance, pairwise cosine, Dirichlet energy, invariant error

### 3.2 结果

| method | depth | energy | v_var | cosine | dirichlet |
|--------|------:|-------:|------:|-------:|----------:|
| GCN | 128 | 0.2003 | **0.0012** | 1.0000 | 0.0000 |
| Residual | 128 | 0.2003 | **0.0012** | 1.0000 | 0.0000 |
| **IsoNode** | 128 | **1.0000** | **1.0000** | **0.0325** | **7940.2** |

**发现**:
- GCN/Residual 在 depth 128 几乎完全 collapse（v_var → 0.0012, cosine → 1.0）
- IsoNode 完美保持 energy=1.0, v_var=1.0
- 归一化 invariant error 仅 1.01e-05，证明 v 方向保持极佳

### 3.3 v-centered variance 的意义

传统 centered variance（投影掉 all-ones 方向）在 GCN depth 128 时为 0.0226，似乎"还有一些方差"。但使用 graph-native 的 v-centered variance（投影掉 v=sqrt(d) 方向）后，collapse 更清晰：**0.0012**，几乎是完全的数值 collapse。

这验证了：对于 symmetric normalized adjacency，v=sqrt(d) 才是自然的不变量方向，而非 all-ones。

---

## 4. Cora Node Classification Downstream 验证

### 4.1 实验设置

- Dataset: Cora (2708 nodes, 1433 features, 7 classes)
- Train/Val/Test: 140/500/2068 (standard split)
- Models: GCN, ResGCN, IsoStreamGCN v2
- Depths: 2, 16, 32
- Epochs: 200, LR: 0.01, Weight decay: 5e-4
- 评估指标: best test accuracy (by validation), final layer variance

### 4.2 单 Seed 结果 (seed=42)

| 模型 | L2 | L16 | L32 |
|------|-----:|-----:|-----:|
| GCN | **77.5%** | 30.7% | 31.2% |
| ResGCN | **75.1%** | 30.0% | 29.6% |
| **IsoStream v2** | — | **69.1%** | **63.9%** |

**Layer Variance (final)**:

| 模型 | L16 | L32 |
|------|-----:|-----:|
| GCN | 0.0000 | 0.0000 |
| ResGCN | 4.9056 | 1.3034 |
| **IsoStream v2** | **4.2732** | **13.7968** |

### 4.3 结果解读

**A. 2-layer baseline 健康**
- GCN L2: 77.5%, ResGCN L2: 75.1%
- 确认训练脚本、数据分割、超参配置正确

**B. Deep GCN 明确 collapse**
- GCN L16/L32: ~30%, variance → 0
- 典型 oversmoothing，分类头无法区分节点

**C. ResGCN 的启示**
- ResGCN L16: variance=4.91 但 accuracy=30.0%
- **关键发现：保持 variance 本身不够**。ResGCN 有非零方差，但 downstream 仍然失败
- IsoStream v2 不只是"让方差不为 0"，而是保留了更有用的深层表示结构

**D. IsoStream v2 显著缓解深层退化**
- L16: 69.1% vs GCN 30.7% = **+130% 提升**
- L32: 63.9% vs GCN 31.2% = **+105% 提升**
- 恢复到了 shallow GCN 性能的 80-90%
- Variance 持续增长（而非 collapse），表示空间保持活跃

**E. Projection 质量**
- orth_error: ~5.6e-07
- fix_error: ~3.6e-07
- 投影精度在训练过程中稳定

### 4.4 v1 vs v2 对比

| 版本 | L16 | L32 | 关键差异 |
|------|-----:|-----:|---------|
| IsoStream v1 | 25.9% | 23.6% | identical streams + mean readout |
| **IsoStream v2** | **69.1%** | **63.9%** | stream_embed + concat readout |

**结论**：之前 GNN downstream 弱是代码/架构设计问题，不是 fixed-vector isometric operator 本身无效。

---

## 5. 多 Seed 稳定性验证

### 5.1 实验设置

- Seeds: [0, 1, 2, 3, 4]
- 配置: GCN L2/L16/L32, ResGCN L2/L16/L32, IsoStream v2 L16/L32
- 报告: mean ± std of best_test_acc

### 5.2 结果

| 配置 | mean ± std | min | max |
|------|-----------:|----:|----:|
| GCN L2 | **77.6% ± 1.9%** | 76.1% | 81.0% |
| ResGCN L2 | **75.8% ± 1.2%** | 74.3% | 77.1% |
| GCN L16 | 31.1% ± 0.2% | 30.9% | 31.5% |
| GCN L32 | 31.2% ± 0.3% | 31.0% | 31.7% |
| ResGCN L16 | 28.8% ± 4.0% | 21.6% | 31.0% |
| ResGCN L32 | 28.6% ± 2.8% | 23.8% | 30.7% |
| **IsoStream v2 L16** | **73.1% ± 1.9%** | 71.5% | 75.8% |
| **IsoStream v2 L32** | **66.6% ± 3.4%** | 61.6% | 71.2% |

### 5.3 稳定性分析

**IsoStream v2 L16 的标准差仅 1.9%**，与 shallow GCN 相同水平，说明结果高度稳定。

对比 ResGCN 深层（std 4.0% / 2.8%），IsoStream v2 的稳定性明显更好。

所有 5 个 seed 中：
- IsoStream v2 L16 始终 > 71%
- IsoStream v2 L32 始终 > 61%
- GCN/ResGCN 深层始终 < 32%

### 5.4 与单 Seed 对比

| 配置 | 单 Seed (42) | 多 Seed (mean) | 差异 |
|------|-------------:|---------------:|-----:|
| GCN L2 | 77.5% | 77.6% | +0.1% |
| ResGCN L2 | 75.1% | 75.8% | +0.7% |
| IsoStream v2 L16 | 69.1% | 73.1% | +4.0% |
| IsoStream v2 L32 | 63.9% | 66.6% | +2.7% |

单 seed=42 恰好是 IsoStream v2 表现较差的 seed，多 seed 均值反而更高。

---

## 6. 关键发现与讨论

### 6.1 机制验证通过

IsoNode 在 synthetic 深度传播中完美保持 energy 和 variance，证明 fixed-vector isometric operator 能防止 graph diffusion collapse。

### 6.2 Downstream 信号出现

IsoStreamGCN v2 在 Cora 上从 deep GCN collapse 的 ~30% 恢复到 64-69%，证明算子的稳定性可以转化为真实的分类性能提升。

### 6.3 ResGCN 的控制实验价值

ResGCN L16 variance=4.91 但 accuracy=30%，说明：
- "保持方差" ≠ "保持有用表示"
- IsoStream 的优势不仅是数值稳定性，而是 stream-zero subspace 提供了额外的结构化信息通道

### 6.4 L32 Variance=13.8 的警示

IsoStream v2 L32 的 final variance=13.80，显著高于 L16 的 4.27。这可能表示：
- isometric path 确实保留了高频信息
- 但也可能存在 representation amplification
- 更深时（L64+）需要调小 beta 或增加 norm 控制

### 6.5 Dense Node-Space Q 的局限性

IsoResGCN（node-space dense Q）验证：
- N=2708 时参数量 1.17 亿，单轮 forward 70 秒
- 不可扩展，不尊重 graph sparsity
- 适合机制 smoke，不适合正式 GNN 架构
- **IsoStream（stream-dimension）是更合理的放置位置**

---

## 7. 代码修复记录

### 7.1 P3: IsoNodeProjection U 预计算改用 fp64

**问题**: `construct_orthogonal_complement_v` 在 fp32 中预计算 U，即使 cast 到 fp64 也无法恢复 U^T v 精度。

**修复**: `IsoNodeProjection.__init__` 中将 v 和 U 注册为 `torch.float64` buffer。

**效果**: n=512 时，U^T v 误差从 2.4e-6 降到 3.4e-15，提升 **7.2 亿倍**。

### 7.2 P1+P2: IsoStreamGCN v2 重构

**问题**:
1. 所有 stream 初始相同（`repeat`）
2. 每个 stream 用同一个 GCN layer，确定性路径完全对称
3. mean readout 因 H·1=1 而抵消 mixing 作用
4. dropout 打在 residual state 上

**修复**:
1. 添加 `stream_embed`（learnable，shape `(s, 1, d)`）
2. dropout 仅作用于 message branch input
3. readout 改为 concat（output dim = s × hidden_dim）

**效果**: accuracy 从 25.9% → 69.1%（L16）

### 7.3 P4: Graph-Native Metrics

**新增**:
- `compute_v_centered_variance(X, X_ref, v)`: 投影掉 v=sqrt(d) 方向后计算 variance
- `compute_invariant_error_norm(X, X_ref, v)`: 归一化 invariant error

**意义**: 对于 symmetric normalized adjacency，v=sqrt(d) 才是自然不变量方向。

---

## 8. 下一步建议

### 8.1 多 Seed 验证（P0）

当前单 seed 结果已显示 strong signal，但需要 5 seeds 确认稳定性。

### 8.2 消融实验（P1）

验证 v2 各项修复的独立贡献：
- v1: identical streams + mean readout
- v2a: stream_embed + mean readout
- v2b: identical streams + concat readout
- v2c: stream_embed + concat readout (full v2)

### 8.3 强 Baseline 对比（P2）

当前 ResGCN 深层仅 30%，可能偏弱。建议添加：
- GCN + PairNorm
- GCNII-style initial residual
- APPNP / SGC

### 8.4 深度稳定性检查（P3）

L32 variance=13.8 需要确认不是 uncontrolled amplification：
- 每层 activation RMS
- 每层 gradient norm
- stream cosine diversity
- 不同 beta 值的影响

### 8.5 更大规模数据集

Cora 仅 2708 节点。下一步应在 Citeseer、PubMed 上验证。

---

## 9. 论文定位

这组结果让 GNN 方向从：

> "机制验证通过，但 downstream 未通过"

变为：

> "初步 downstream 通过，需要多 seed 和强 baseline 确认"

最重要的意义：
1. 证明之前 GNN downstream 弱不是算子无效
2. 证明代码/架构修复后，下游准确率立刻恢复
3. 证明 fixed-vector isometric stream path 能实际缓解 deep GNN collapse
4. 让 "Iso 算子是通用 neural mixing primitive" 的叙事更可信

---

## 附录 A: 文件清单

| 文件 | 说明 |
|------|------|
| `gnn/projection.py` | IsoNodeProjection（fp64 U 预计算） |
| `gnn/models.py` | GCN, ResGCN, IsoStreamGCN v2, IsoResGCN |
| `gnn/utils.py` | v-centered variance, normalized invariant error |
| `experiments/gnn_stage1_synthetic_oversmoothing.py` | Synthetic oversmoothing (G1) |
| `experiments/gnn_stage1_cora_isores.py` | Cora 对比实验 (G4) |
| `results/gnn_stage1/synthetic_v2/` | Synthetic 原始结果 |
| `results/gnn_stage1/cora_isores_fast/` | Cora 单 seed 原始结果 |
