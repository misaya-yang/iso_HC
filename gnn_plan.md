# IsoHC / Fixed-Vector Isometric Operator 的 GNN 一小时验证计划（单卡 RTX 5090）

## 结论先行

可以。
但 **1 小时内能验证的是机制可行性，不是完整 GNN 论文结论**。

这一阶段的目标不是证明 IsoGNN SOTA，而是快速回答：

1. 这个 fixed-vector isometric operator 能不能自然放进 GNN setting；
2. 它是否能缓解 repeated graph propagation 的能量衰减 / oversmoothing；
3. 它在 Cora 级别小图上是否能稳定训练；
4. 是否值得进入第二阶段做严肃 GNN 实验。

---

# 0. 核心思想

标准 GNN propagation 通常类似：

[
X_{l+1}=SX_l
]

其中 (S) 是归一化 adjacency，例如：

[
S=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}
]

这种传播是扩散型的。深层后容易出现：

[
X_l \rightarrow \text{low-frequency / constant subspace}
]

也就是 oversmoothing。

你的 Iso 算子提供另一种传播方式：

[
Q\in \mathcal M_v={Q\in O(n):Qv=v}
]

它满足：

[
Q^\top Q=I
]

[
Qv=v
]

如果用于 graph node dimension，它能保持某个 graph-invariant direction，同时不耗散节点特征能量。

对 symmetric normalized adjacency：

[
S=\tilde D^{-1/2}\tilde A\tilde D^{-1/2}
]

自然的不变量不是普通 (\mathbf 1)，而是：

[
v=\sqrt{\tilde d}
]

因为：

[
S\sqrt{\tilde d}=\sqrt{\tilde d}
]

所以 GNN 版 fixed-vector operator 应优先使用：

[
\mathcal M_{\sqrt d}
====================

{Q\in O(N):Q\sqrt d=\sqrt d}
]

这比直接用 (\mathbf 1) 更严谨。

---

# 1. 一小时验证的整体结构

分三步跑。

| 阶段 |                                       内容 |        时间 |
| -- | ---------------------------------------: | --------: |
| G1 | Synthetic graph oversmoothing diagnostic | 10–15 min |
| G2 |  Dense node-space IsoProp mechanism test | 10–15 min |
| G3 |            Cora IsoStream-GNN smoke test | 25–30 min |
| 汇总 |                         结果表与 go/no-go 判断 |     5 min |

总计约 50–60 分钟。

---

# 2. 本阶段不要做什么

一小时内不要做：

* OGB；
* 大图；
* GraphGPS / GAT / GraphSAGE 大量对照；
* 多 seed 完整统计；
* 追求 Cora SOTA；
* 复杂 hyperparameter sweep；
* 大规模 dense (N\times N) projection。

本阶段只做机制验证。

---

# 3. 两条 GNN 路线

## 路线 A：Node-space IsoProp

这是最直接的数学验证。

图有 (N) 个节点，特征为：

[
X\in\mathbb R^{N\times d}
]

构造：

[
Q=\Pi_v(A)
]

其中：

[
v=\sqrt{\tilde d}
]

[
Qv=v,\qquad Q^\top Q=I
]

然后比较：

[
X_{l+1}=SX_l
]

和：

[
X_{l+1}=QX_l
]

这直接验证：

* diffusion propagation 是否收缩；
* isometric propagation 是否保持 feature energy；
* graph DC component 是否保持；
* oversmoothing 是否被避免。

缺点：(Q) 是 dense (N\times N)，不能直接用于大图训练。
所以这里只能在 (N=256) 或 (N=512) synthetic graph 上做机制验证。

---

## 路线 B：Stream-space IsoGNN

这是更接近实际 GNN layer 的版本。

不在 node dimension 上做 (N\times N) dense projection，而是在小的 stream dimension 上做：

[
H\in\mathbb R^{s\times s}
]

[
H^\top H=I,\qquad H\mathbf 1=\mathbf 1
]

其中 (s=4) 或 (s=8)。

每个节点维护多个 propagation stream：

[
X_l\in\mathbb R^{s\times N\times d}
]

每层做：

[
X_{l+1}=H_lX_l+\beta \cdot \mathrm{GNNBlock}(X_l)
]

这和 IsoHC 最接近，但 substrate 换成了 GNN。
它的优点是便宜、可训练、能直接在 Cora 上跑。

一小时验证建议两条都跑：

* Synthetic 用路线 A；
* Cora 用路线 B。

---

# 4. 实验 G1：Synthetic Graph Oversmoothing Diagnostic

## 目的

先不用训练，只看 repeated graph propagation 的机制。

比较：

[
X_{l+1}=SX_l
]

是否导致特征方差和高频能量快速衰减。

---

## 数据

用 synthetic stochastic block model。

```yaml
graph:
  type: two_block_sbm
  num_nodes: 512
  num_classes: 2
  p_in: 0.06
  p_out: 0.01
  self_loops: true

features:
  dim: 64
  class_signal: 1.0
  noise_std: 0.5
```

生成节点标签 (y_i\in{0,1})。
特征初始化：

[
X_i = \mu_{y_i} + \epsilon_i
]

其中：

[
\epsilon_i\sim\mathcal N(0,\sigma^2 I)
]

---

## 方法

### Method 1: GCN diffusion

[
X_{l+1}=SX_l
]

### Method 2: residual diffusion

[
X_{l+1}=(1-\alpha)SX_l+\alpha X_l
]

建议：

```yaml
alpha: 0.1
```

### Method 3: IsoNode propagation

构造：

[
Q=\Pi_v(S+\epsilon E)
]

其中：

[
v=\sqrt{\tilde d}
]

(E) 是小的 random perturbation，用来避免 (\Pi_v(S)) 在某些图上退化得过于接近 identity。

建议：

```yaml
epsilon: 0.02
```

然后：

[
X_{l+1}=QX_l
]

---

## 深度

```yaml
depths: [1, 2, 4, 8, 16, 32, 64, 128]
```

---

## 记录指标

### 1. Feature energy ratio

[
r_l=\frac{|X_l|_F}{|X_0|_F}
]

GCN diffusion 通常下降。
IsoNode 应接近 1。

---

### 2. Centered feature variance

定义：

[
\bar X_l=\frac{1}{N}\mathbf 1\mathbf 1^\top X_l
]

[
V_l=\frac{|X_l-\bar X_l|_F}{|X_0-\bar X_0|_F}
]

oversmoothing 时：

[
V_l\rightarrow 0
]

IsoNode 应该显著更高。

---

### 3. Graph Dirichlet energy

[
E_l=\operatorname{tr}(X_l^\top L X_l)
]

其中：

[
L=I-S
]

GCN diffusion 会快速降低 Dirichlet energy。
IsoNode 不一定保持 Dirichlet energy，但不应像 diffusion 那样快速坍缩。

---

### 4. Mean / invariant error

对 (v=\sqrt{\tilde d})：

[
m_l=
\left|
v^\top X_l-v^\top X_0
\right|_2
]

IsoNode 应接近 0。

---

### 5. Pairwise cosine collapse

随机采样节点对：

[
C_l=\mathbb E_{i\ne j}
\frac{\langle X_{l,i},X_{l,j}\rangle}{|X_{l,i}||X_{l,j}|}
]

oversmoothing 时 (C_l) 会升高接近 1。

---

## 通过标准

G1 通过条件：

```text
At depth 64 or 128:

GCN diffusion:
  centered variance ratio drops below 0.2
  pairwise cosine increases substantially

IsoNode:
  feature energy ratio in [0.98, 1.02]
  invariant error < 1e-5 in fp32 or < 1e-10 in fp64
  centered variance ratio remains much higher than GCN
```

注意：
如果 IsoNode 的 centered variance 保持过高，这不是坏事。它说明 isometry 确实避免了 diffusion collapse。后续任务性能需要用训练实验判断。

---

# 5. 实验 G2：Dense Node-Space IsoProp Classification Probe

## 目的

在同一个 synthetic graph 上加一个极简分类 probe，看 IsoNode 是否至少不破坏 class signal。

这不是正式 GNN benchmark，只是快速 sanity check。

---

## 设置

先传播 (L) 层，得到 (X_L)。
然后训练一个 linear classifier：

[
\hat y = \operatorname{softmax}(X_L W)
]

只训练 (W)，不训练 propagation。

---

## 方法

```yaml
methods:
  - raw_features
  - gcn_diffusion
  - residual_diffusion
  - isonode
depths:
  - 8
  - 16
  - 32
  - 64
```

---

## 训练

```yaml
classifier:
  epochs: 100
  lr: 0.01
  weight_decay: 5e-4
  optimizer: adam
```

数据 split：

```yaml
train_per_class: 20
val_per_class: 50
rest: test
```

---

## 通过标准

```text
At depth 32 or 64:

GCN diffusion accuracy should degrade or variance collapse.
IsoNode should retain usable class signal better than deep diffusion.
```

不要求 IsoNode 一定最高。
只要它明显避免深层 collapse，就值得进入下一阶段。

---

# 6. 实验 G3：Cora IsoStream-GNN Smoke Test

## 目的

验证你的算子能否作为 GNN layer 的一部分稳定训练。

这里不做 dense (N\times N) node-space projection，而是做 stream-space Iso mixing。

---

## 数据

优先使用 Cora。

```python
torch_geometric.datasets.Planetoid(root="data", name="Cora")
```

如果服务器没有 PyG 或 Cora 缓存，跳过 G3，不阻塞本阶段结论。
G1/G2 已经能完成机制验证。

---

## 模型 1：标准 GCN baseline

[
X_{l+1}=\sigma(SX_lW_l)
]

配置：

```yaml
model: GCN
hidden_dim: 64
layers: [2, 8, 16, 32]
dropout: 0.5
```

---

## 模型 2：Deep residual GCN

[
X_{l+1}=X_l+\beta\sigma(SX_lW_l)
]

配置：

```yaml
model: ResGCN
hidden_dim: 64
layers: [16, 32]
beta: 0.5
dropout: 0.5
```

---

## 模型 3：IsoStream-GCN

维护 (s) 个 streams：

[
X_l\in\mathbb R^{s\times N\times d}
]

每层：

[
Y_{l,s}=\sigma(SX_{l,s}W_l)
]

然后：

[
X_{l+1}
=======

H_lX_l+\beta Y_l
]

其中：

[
H_l\in\mathcal M_{\mathbf 1_s}
]

即：

[
H_l^\top H_l=I
]

[
H_l\mathbf 1_s=\mathbf 1_s
]

最后输出用 stream mean：

[
Z_l=\frac{1}{s}\sum_{i=1}^s X_{l,i}
]

分类：

[
\hat y=\operatorname{softmax}(Z_L W_{\mathrm{out}})
]

---

## IsoStream 配置

```yaml
model: IsoStreamGCN
streams: 4
hidden_dim: 64
layers: [16, 32]
beta: 0.5
ns_steps: 5
projection_dtype: fp32
dropout: 0.5
```

初始化建议：

```text
H_raw initialized near identity
```

例如：

[
\tilde H=I+0.01\epsilon
]

这样前期不会因为 stream mixing 太强导致训练不稳。

---

## 训练配置

```yaml
training:
  epochs: 200
  lr: 0.01
  weight_decay: 5e-4
  optimizer: adam
  early_stop: false
  precision: fp32
```

Cora 很小，fp32 足够。
不建议一开始用 bf16，因为这个实验要看数值稳定性。

---

## 记录指标

### 训练指标

```text
train loss
train accuracy
val accuracy
test accuracy
grad norm
NaN / Inf count
```

### Oversmoothing 指标

每隔 10 epoch 记录每层 hidden representation 的：

#### Centered variance

[
V_l=\frac{|X_l-\bar X_l|_F}{|X_0-\bar X_0|_F}
]

#### Pairwise cosine

[
C_l=\mathbb E_{i\ne j}
\frac{\langle X_{l,i},X_{l,j}\rangle}{|X_{l,i}||X_{l,j}|}
]

#### Dirichlet energy

[
E_l=\operatorname{tr}(X_l^\top L X_l)
]

### IsoStream 专属指标

[
\epsilon_{\mathrm{orth}}=|H_l^\top H_l-I|_F
]

[
\epsilon_{\mathrm{fix}}=|H_l\mathbf 1-\mathbf 1|_2
]

[
r_{\mathrm{stream}}=
\frac{|H_lX_l|_F}{|X_l|_F}
]

stream diversity：

[
D_l=
\frac{1}{s(s-1)}
\sum_{i\ne j}
\cos(X_{l,i},X_{l,j})
]

---

## 通过标准

G3 通过条件：

```text
IsoStream-GCN 16-layer:
  finishes 200 epochs
  no NaN / Inf
  max orth_error < 1e-5
  max fix_error < 1e-5
  stream energy ratio in [0.98, 1.02]
  val accuracy not catastrophically worse than deep GCN
```

更强的通过条件：

```text
IsoStream-GCN 32-layer:
  trains stably
  hidden variance collapse is weaker than standard 32-layer GCN
  pairwise cosine is lower than standard deep GCN
```

不要求一小时内打败所有 baseline。
只要求证明“可训练 + 不塌缩 + 几何约束有效”。

---

# 7. 推荐命令结构

目录建议：

```text
/root/isoHC/
  experiments/
    gnn_stage1_synthetic_oversmoothing.py
    gnn_stage1_synthetic_probe.py
    gnn_stage1_cora_isostream.py
  results/
    gnn_stage1/
      synthetic/
      cora/
```

---

## 命令 1：Synthetic oversmoothing

```bash
python experiments/gnn_stage1_synthetic_oversmoothing.py \
  --num-nodes 512 \
  --feature-dim 64 \
  --p-in 0.06 \
  --p-out 0.01 \
  --depths 1 2 4 8 16 32 64 128 \
  --methods gcn residual isonode \
  --dtype fp32 \
  --disable-tf32 \
  --device cuda \
  --out results/gnn_stage1/synthetic
```

如果 fp32 invariant error 边界偏高，补跑：

```bash
python experiments/gnn_stage1_synthetic_oversmoothing.py \
  --num-nodes 512 \
  --feature-dim 64 \
  --p-in 0.06 \
  --p-out 0.01 \
  --depths 64 128 \
  --methods isonode \
  --dtype fp64 \
  --disable-tf32 \
  --device cuda \
  --out results/gnn_stage1/synthetic_fp64
```

---

## 命令 2：Synthetic classifier probe

```bash
python experiments/gnn_stage1_synthetic_probe.py \
  --num-nodes 512 \
  --feature-dim 64 \
  --p-in 0.06 \
  --p-out 0.01 \
  --depths 8 16 32 64 \
  --methods raw gcn residual isonode \
  --epochs 100 \
  --device cuda \
  --out results/gnn_stage1/synthetic_probe
```

---

## 命令 3：Cora IsoStream smoke

```bash
python experiments/gnn_stage1_cora_isostream.py \
  --dataset Cora \
  --models gcn resgcn isostream \
  --layers 16 32 \
  --hidden-dim 64 \
  --streams 4 \
  --epochs 200 \
  --ns-steps 5 \
  --projection-dtype fp32 \
  --device cuda \
  --out results/gnn_stage1/cora
```

---

# 8. 一小时内建议只跑这些配置

为了控制时间，不要一开始全扫。

## 必跑配置

```yaml
synthetic:
  num_nodes: 512
  feature_dim: 64
  depth: [16, 32, 64, 128]
  methods: [gcn, residual, isonode]

cora:
  layers: [16, 32]
  hidden_dim: 64
  streams: 4
  methods: [gcn, resgcn, isostream]
  epochs: 200
```

## 不跑

```yaml
skip:
  - Citeseer
  - PubMed
  - OGB
  - GAT
  - GraphSAGE
  - hidden_dim sweep
  - stream sweep
  - seed sweep
```

---

# 9. 输出图表

一小时验证至少产出 4 张图。

## Figure 1：Synthetic energy across depth

```text
x-axis: depth
y-axis: ||X_l||_F / ||X_0||_F
lines: GCN, Residual GCN, IsoNode
```

预期：

```text
GCN declines
Residual declines slower
IsoNode stays near 1
```

---

## Figure 2：Synthetic centered variance across depth

```text
x-axis: depth
y-axis: ||X_l - mean(X_l)|| / ||X_0 - mean(X_0)||
lines: GCN, Residual GCN, IsoNode
```

预期：

```text
GCN collapses
IsoNode preserves variance
```

---

## Figure 3：Synthetic pairwise cosine across depth

```text
x-axis: depth
y-axis: average pairwise cosine
lines: GCN, Residual GCN, IsoNode
```

预期：

```text
GCN approaches high cosine
IsoNode remains lower
```

---

## Figure 4：Cora hidden variance by layer

```text
x-axis: layer
y-axis: centered hidden variance
lines: 16/32-layer GCN, ResGCN, IsoStream-GCN
```

预期：

```text
Deep GCN variance collapses faster
IsoStream-GCN retains more variance
```

---

# 10. 结果汇总模板

实验结束后用这个表判断。

```markdown
# GNN Stage 1 Summary

## Synthetic Oversmoothing

| method | depth | energy ratio | centered variance ratio | pairwise cosine | invariant error | pass |
|---|---:|---:|---:|---:|---:|---|
| GCN | 128 | | | | n/a | |
| Residual | 128 | | | | n/a | |
| IsoNode | 128 | | | | | |

## Synthetic Probe

| method | depth | test acc | centered variance ratio | pass |
|---|---:|---:|---:|---|
| Raw | 0 | | | |
| GCN | 32 | | | |
| GCN | 64 | | | |
| IsoNode | 32 | | | |
| IsoNode | 64 | | | |

## Cora Smoke

| model | layers | final val acc | test acc | max orth err | max fix err | variance collapse? | pass |
|---|---:|---:|---:|---:|---:|---|---|
| GCN | 16 | | | n/a | n/a | | |
| GCN | 32 | | | n/a | n/a | | |
| ResGCN | 32 | | | n/a | n/a | | |
| IsoStream | 16 | | | | | | |
| IsoStream | 32 | | | | | | |

## Decision

- [ ] GNN Stage 1 Pass
- [ ] Partial Pass: mechanism works, training needs adjustment
- [ ] Fail: operator does not show useful GNN signal

Notes:
-
-
-
```

---

# 11. 通过 / 失败怎么解释

## 最理想结果

```text
Synthetic:
  GCN energy and variance collapse.
  IsoNode preserves energy and variance.

Cora:
  IsoStream trains stably at 16/32 layers.
  IsoStream has weaker hidden collapse than deep GCN.
  Projection errors remain tiny.
```

解释：

> fixed-vector isometric operator has a plausible GNN use case: it can preserve graph-level invariant components while avoiding diffusion-driven oversmoothing.

这足够支持进入第二阶段。

---

## 部分通过

可能结果：

```text
Synthetic clearly passes.
Cora IsoStream trains, but accuracy not better.
```

解释：

> 算子机制成立，但 IsoStream-GNN architecture 还需要设计。
> 不否定 GNN 方向。

下一步应改：

* residual coefficient；
* stream initialization；
* normalization；
* readout；
* 是否使用 Jumping Knowledge；
* 是否把 Iso operator 放在 propagation 前还是后。

---

## 失败情况 1：Synthetic IsoNode 不保持 invariant

优先检查：

1. (v=\sqrt{\tilde d}) 是否正确；
2. (v) 是否归一化；
3. (U^\top v=0) 是否满足；
4. projection 是否使用 fp32 / fp64；
5. TF32 是否关闭；
6. (Qv-v) 是每层误差还是累积误差。

---

## 失败情况 2：IsoNode 只是 identity

如果：

[
\frac{|Q-I|_F}{\sqrt N}<0.01
]

说明：

[
\Pi_v(S)
]

太接近 identity，机制测试不够强。

处理：

```text
Use A = S + epsilon * edge_masked_noise
epsilon in {0.02, 0.05, 0.1}
```

同时记录：

```text
||Q - I||_F / sqrt(N)
```

IsoNode 必须既保持等距，又有非平凡 mixing。

---

## 失败情况 3：Cora IsoStream accuracy 很差

不要马上否定。优先判断是否是 architecture 问题。

检查：

1. 2-layer baseline 是否正常；
2. IsoStream 是否用了过强 stream mixing；
3. (H_{\mathrm{raw}}) 是否 near-identity 初始化；
4. residual coefficient (\beta) 是否过大；
5. dropout 是否过高；
6. stream readout 是否合理；
7. 是否需要 layer norm / pair norm。

建议先试：

```yaml
beta: [0.1, 0.3, 0.5]
H_init: identity_plus_noise
streams: 4
layers: 16
```

---

# 12. 一小时验证的 go/no-go 标准

## Go

满足以下条件就继续 GNN 方向：

```text
Synthetic IsoNode:
  energy ratio at depth 128 in [0.98, 1.02]
  invariant error small
  centered variance much higher than GCN

Cora IsoStream:
  16-layer model trains without NaN
  projection errors stay small
  hidden variance collapse weaker than deep GCN
```

## No-Go

同时出现以下情况才暂停：

```text
Synthetic IsoNode fails energy/invariant preservation.
Cora IsoStream cannot train even at 16 layers.
Projection errors are large despite fp32/fp64.
```

如果只是 Cora accuracy 不高，不算 No-Go。

---

# 13. 第二阶段如果 GNN 方向通过

通过后再规划更正式的 GNN 实验：

1. 多 seed：5 或 10 seeds；
2. 数据集：Cora / Citeseer / PubMed；
3. heterophily：Chameleon / Squirrel / Actor；
4. long-range graph benchmark；
5. 对照：GCNII、APPNP、PairNorm、DropEdge、GraphNorm；
6. 更严谨的 graph-local isometric operator；
7. generalized weighted isometry：

[
Q^\top GQ=G,\qquad Qv=v
]

对于 random-walk GNN，可能需要：

[
G=D
]

这会比当前 Euclidean isometry 更 graph-native。

---

# 14. 最关键的判断

这套一小时实验如果跑通，你可以得到一个清晰结论：

> IsoHC 的 fixed-vector isometric projection 不只是 HC residual stream trick；它在 GNN 的 repeated propagation / oversmoothing 场景中也有明确机制信号。

这就是下一篇 paper 冲高的第一块证据。
