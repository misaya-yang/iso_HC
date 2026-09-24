# IsoHC 研究方向重评估（Salvage Investigation）

**日期：** 2026-09-24　**性质：** 研究判断，不是论文草稿；`paper/` 未改动。
**证据等级约定：** 【实测】本仓库数据或本次复现；【推导】本次给出证明/数值验证的理论结论；【文献-全文】通过 GitHub 镜像读过全文；【文献-摘要】仅摘要/检索摘要（arXiv、OpenReview、Semantic Scholar、HuggingFace 在本环境被网络策略屏蔽）。

---

## 0. 结论先行

1. **IsoHC 作为"方法"应当终止。** 静态、保均值（`Q1=1, 1ᵀQ=1ᵀ, QᵀQ=I`）的 stream transport 与 identity transport 是**精确的 gauge 等价**：同一函数类、同一 read/write 参数类、零条件数代价（float64 下 logits 差 3e-16）【推导】。它在仓库里 ≈ identity-HC【实测】，在 oHC（Baidu, 3.9B）的干预实验里也被预测 ≈ iHC【文献-全文】。Birkhoff 收缩与正交 HC 两个卖点均已被 2026 年的工作覆盖。
2. **我们过去没看到的东西：IsoHC 要保护的 invariant 恰恰是让额外 streams 失效的原因。** 保均值 ⇒ 在 `[mean | 1^⊥]` 基下 `H = diag(1, B)`：主通道与 complement 被 transport 完全解耦，complement 只能经由有界的 read/write 接触 loss，uniform depth kernel 被钉死为全 1【推导】。仓库里唯一改善 loss 的 variant（unconstrained, `H1≠1`）、GNN 单 seed 里最好的 orthogonal-only、以及 3.9B 规模上显著胜出的 oHC，**全都打破了这个 invariant**；oHC 的干预把增益定位到 mean↔difference 交换通道，并明确说 mean-preserving SO(n−1) 对照臂（=IsoHC）"was not trained"【文献-全文】。
3. **仓库里最强的"机制结果"其实不是学到的。** 48L/24L 训练后 mHC 的 1^⊥ 谱（0.92221 / 0.91563）只用 init（`diag_bias=4`）+ AdamW decoupled weight decay 重放 LR schedule 即可预测（0.92174 / 0.91542）；composite 4.0e-4 vs 实测 4.39e-4【实测+推导】。loss 对 transport 的梯度可以忽略——这是"complement 没被用"的定量证据。
4. **唯一值得继续的路线：** 把项目从"等距 HC 方法"转为 **"hyper-connections 的 gauge 理论 + 等距主通道–补空间交换（isometric exchange）机制"**。核心主张：静态 transport 是纯 gauge；mixing 的物理内容只有 (i) token 依赖（curvature）、(ii) 相对 read/write 约束的破缺（mean↔complement exchange，功能上是"无损的、token 选择性的主通道 depth 重加权"）、(iii) 权重共享循环中的 holonomy；dissipation 只改变可达性/条件数。IsoHC 在这条线上是**最关键的对照臂**。
5. **先做 3–4 周决定性筛选再决定是否投入**（~100–150M，8 个 dynamic 臂 + 单 stream 可学习 residual 权重对照，关键臂 2 seeds），kill 条件见 §8。CPU 小模型上等距交换没有可检测收益，唯一稳健的杠杆是非等距的"前几层放大"主通道增益（与 LAuReL-RW 重合），所以我对筛选通过的主观概率只有 ~25–35%。若不通过，把 gauge 定理 + WD artifact + 训练过的 SO(n−1) 对照写成短文（workshop/TMLR）后停止。

---

## 1. 这次调查做了什么

- 通读 `lm/`、`isohc/`、`experiments/` 全部代码，`docs/0605_alldoc` 下全部报告与 raw JSON，`docs/0714`、`docs/0715`、superpowers 计划与 git 历史（28 commits，2026-05-11 → 07-15）。
- 两路并行文献调查（HC 家族；depth routing / residual 理论），覆盖到 2026-09 的约 60 篇工作；oHC、xHC、Qwen3.8-Next、Kimi AttnRes、DDL、Zwick C-mHC、DeepSeek-V4/GLM-5.3/Hy4 代码读了全文（本地副本未入库）。
- 便宜诊断（全部 CPU，无 GPU 训练）：
  - `scripts/wd_attribution.py`：重放两次训练的 LR schedule，只施加 weight decay。
  - `scripts/gauge_check.py`、`scripts/gauge_table.py`：在仓库的 `TwoBranchHCTransformer` 上数值验证 gauge 等价（float64）。
  - `scripts/access_probe.py`：从未运行过的 P1 "accessibility" 问题的 CPU 小模型版（16 层、d=64、char-level TinyShakespeare、1200 步；只作方向性证据）。结果见附录 B。

---

## 2. Q1：截至今天真正成立的 scientific contribution

| 等级 | 结论 | 说明 |
|---|---|---|
| A 可靠 | Birkhoff 非扩张；严格正 ⇒ 1^⊥ 严格收缩；`B_n ∩ O(n)` = permutation；fixed-vector 正交保 mean 与 1^⊥ 范数 | 数学正确，但属 Perron–Frobenius / Dobrushin 经典结论的直接应用；2026 年已被 Zwick C-mHC（显式速率）、oHC 2609.02672、Homogeneity Trap 2601.02080、JPmHC 2602.18308 独立发表 |
| A 可靠 | NS fixed-vector projection 实现正确 | fix error ~3e-7，orth error ~5e-4（NS-5, n=4） |
| A 可靠 | 静态 Sinkhorn proxy 48L/96 transports composite 1^⊥ gain 4.39e-4 vs IsoHC 0.996 | 数值正确 |
| B 成立但含义不同 | 上述收缩**不是训练学到的** | init + WD 预测到 4 位有效数字（§3.2） |
| B 成立但含义不同 | IsoHC ≈ identity-HC | 48L FE 5.7206 vs 5.7157；MPS 125M 6.4402 vs 6.4426；Q→I 替换 +7e-4。现在有理论解释（§3.3），并恰是 oHC 缺的对照 |
| C proxy/toy | 1024 层 residual-only detector | 无 block，结果由构造决定 |
| C proxy/toy | GNN 深层 Cora 恢复 | 收益来自 stream_embed + concat readout；H 类型消融单 seed、噪声极大（逻辑相同的 none/identity 在 L32 相差 14 点）；"iso 在 L32 优于 identity"的多 seed 数字不在仓库里 |
| C 未回答核心问题 | 所有 LM loss 比较 | 1 seed；20M tokens（48L 177M 参数，T/P≈0.11）；WD 施加于全部参数；λ 从未记录；mHC 是静态 proxy |
| D 本次新增 | static gauge 定理、coupling calculus、WD artifact、"invariant 即瓶颈" | §3 |

---

## 3. Q2：为什么没有转化成足够强的方法论文

### 3.1 实验在结构上看不到差异（coupling）【推导+实测】

仓库的 read/write：`a = 1 + λ_a P⊥w`，`b = 1 + λ_b P⊥v`，λ 是全局共享标量、init 0.01，且与所有参数一起被 WD 0.1 拉向 0。取 `X = 1μᵀ + UC`：

```
μ_{k+1} = μ_k + y_k                 （精确的标准 Pre-LN residual）
C_{k+1} = B_k C_k + β_k y_kᵀ,   z_k = μ_k + α_kᵀ C_k,   |α|≈0.004, |β|≈0.017
```

对任何保均值的静态 transport，depth kernel（block k 的输入中 block j 输出的系数）为

```
M_kj = 1 + (λ_a λ_b / n) · w_kᵀ P⊥ Φ_{k,j+1} P⊥ v_j
```

即"全 1 下三角（标准 residual）+ 1e-4 量级扰动"。complement removal / replacement 只有 1e-4–1e-3 影响是**必然结果**，不是 hypothesis 的反证。CPU probe 旁证：去掉 HC 参数上的 WD 后，|λ| 在 1200 步内从 0.01 大多长到 0.2–0.5。

### 3.2 Transport 从未被 loss 训练（WD artifact）【实测+推导】

`scripts/wd_attribution.py` 的输出：

| run | 步数 | WD 对 logits 的累计因子 | WD-only 预测 1^⊥ sv (mean/min/max) | 训练后实测 | composite 预测 / 实测 |
|---|---:|---:|---|---|---|
| 48L FineWeb-Edu | 1954 | 0.9685 | 0.92174 / 0.92021 / 0.92328 | 0.92221 / 0.92075 / 0.92373 | 4.0e-4 / 4.39e-4 |
| 24L TinyStories | 3256 | 0.9481 | 0.91542 / 0.91389 / 0.91710 | 0.91563 / 0.91302 / 0.91842 | 1.4e-2 / — |

初始 0.93055 → 训练后的"收缩加深"完全是 decoupled WD 把 Sinkhorn logits 拉向 0（= 均匀混合、最大熵）。推论：
- "mHC 在训练中学会了收缩"不成立；loss 对 H 几乎无梯度。
- WD 对 Sinkhorn logits 是一个**熵正则**，其不动点是 `11ᵀ/n`（完全同质化）；在 ≥100K 步的正常预训练里，若 loss 不主动抵抗，logits 会衰减到使每步保留 <25% complement 的水平。这是一个与几何无关、却会制造"收缩病理"的优化器机制——任何比较 Birkhoff 与其它 mixer 的实验都必须报告 mixing 参数是否被 WD。
- CPU probe 中即使不加 WD、λ 长到 ~0.4，mHC 的 1^⊥ 谱仍停在初值（0.929 vs 0.9306）：loss 不"反抗"收缩。

### 3.3 更根本：静态 IsoHC 不可能优于 identity-HC（gauge 定理）【推导+数值验证】

**定理（静态 gauge）。** 令 `X_k = G_k Z_k`，`G_0 = I`，`G_{k+1} = H_k G_k`。任意静态可逆 transport 序列的网络，与 identity transport、read `ã_k = G_kᵀ a_k`、write `b̃_k = G_{k+1}^{-1} b_k`、exit `ã_out = G_Lᵀ a_out` 的网络逐点相同（block 的 RMSNorm 输入 `a_kᵀX_k` 不变）。

推论取决于 read/write 参数类是否在 gauge 作用下封闭：

| 静态 transport | ≡ identity-HC（自由 read/write） | 仍在仓库参数类 `1ᵀv=n` 内 | 等价 write 向量范数代价 |
|---|---|---|---|
| IsoHC（SO(n−1)，保均值） | 是（logit 差 3.3e-16） | **是** | 1.00× |
| Birkhoff（mHC，可逆） | 是（2.1e-15） | 是 | **194×**（测试用 12 transports、diag_bias 2；随深度按 ρ^{-k} 增长，按 48L 实测 composite 4.39e-4 约为 2300×） |
| 全正交 O(n)（oHC 型） | 是（3.1e-16） | **否**（均值权重被改写） | 1.00× |
| unconstrained | 是（2.2e-9） | 否 | 2.4e7× |

（`scripts/gauge_table.py`，6 层 × 2 branch，大角度随机 transport。）

定理只覆盖**静态** transport：token 依赖的 transport 不能被静态 read/write 吸收（这正是下文"curvature"的含义），所以动态情形下的结论（如 dyn-IsoHC ≈ iHC）是由解耦结构 + oHC 干预给出的**预测**，需要 §7 的训练对照来检验。

含义：
- **静态 IsoHC = identity-HC 的零代价重参数化**：在仓库参数类里不可能带来表达力，只可能带来优化效应；而实测吞吐只有 identity-HC 的 65%。
- **静态 Birkhoff = identity-HC 的病态重参数化**：它的"坏处"是把长程 complement kernel 变得指数昂贵（局部化偏置）；IsoHC 的"好处"只是撤销这个偏置——identity 也能做到。
- 仓库的主要诊断（composite 1^⊥ gain、stream cosine）描述的是**坐标选择/条件数**而不是函数（oHC 也指出单个 stream cosine 不是不变量）。这正是"4e-4 vs 0.996 的强机制结果没有 loss 后果"的原因。
- 对 mHC/oHC 这类 **σ-有界非负** read/write，gauge 作用不封闭：静态 mixing 的作用变成"让受限的 read/write 够到原本够不到的方向"。这预测：read/write 越富（有符号、逐通道），H_res 越冗余——与 Qwen3.8-Next（25B-A3B/560B tokens：去掉 H_res 的 Gated Residual 1.590 优于 dynamic mHC 1.594）和 Tencent Hy4（iHC，无 mixing）的生产级选择一致【文献-全文】。

### 3.4 我们过去没看到的：被保护的 invariant 才是瓶颈【推导+实测+文献】

保均值（`H1=1` 且 `1ᵀH=1ᵀ`）⇒ `Eᵀ H E = [[1, 0],[0, B]]`（`E=[1/√n | U]`）。于是：
- transport 永远不会把主通道内容移入 complement，也不会把 complement 移回主通道；complement 只能通过 `α, β` 影响输出（§3.1 的 O(λ_aλ_b) 耦合）。
- uniform depth kernel 被钉死为全 1：每层都必须以权重 1 看到所有之前层的输出，模型无法做**与距离相关的重加权**（recency），除非 read/write 的均值分量随深度指数变化（有界参数化里不可能）。

mHC 与 IsoHC 都施加这个约束。证据指向打破它的一侧：
- **仓库**：唯一改善 loss 的是 unconstrained（fix error 0.105/0.149）：48L FE 5.676 vs 5.716–5.725；24L TS 2.196 vs 2.249–2.260（各 1 seed，两次同号）。WD 单独就会把它的 mean gain 拉到 ~0.97/0.95 每 transport；fix error `|H1−1|` 不带符号，所以梯度把 gain 推向哪一侧从现有日志无法判断（orth error 0.17–0.26 说明还有其他偏离）。
- **GNN 单 seed**：orthogonal-only（不固定 1）L16 76.8% 最好（iso 72.5%，identity 70.1%），当时被当作"偏离 uniform"的警告。仓库里 `OrthogonalMixing` 一直存在，但被刻意排除在主 runner 之外。
- **oHC（2609.02672）**：3.9B-A0.4B MoE、73B tokens、Muon；16 项下游平均 BPB（seed σ=0.00646）：baseline 1.0920、mHC 1.0802、iHC 1.0720、oHC（SO(4)，不保均值）1.0531。oHC 胜 baseline 6.02σ、胜 mHC 4.20σ、胜 iHC 2.94σ；mHC 胜 baseline 仅 1.82σ、iHC 胜 mHC 1.26σ，均不显著。把训练好的 oHC 的两个 cross 块（mean↔difference）置零后与 identity 不可区分；在 1^⊥ 内把旋转角放大到 2.5 rad（即 IsoHC 方向）落在 identity 的 1.8% 以内。作者明确写 mean-preserving SO(3) 臂 "was not trained"。
- **生产消融**：Qwen3.8、Hy4 放弃 H_res；DeepSeek-V4-Flash 的分析（2609.05309【文献-摘要】）显示后层 mixing 接近 identity、每个 read/write 位点有效只用 ~2 个 stream。

**功能解释（本报告提出的假设）：** 保均值的真正代价是**主通道 depth kernel 的 uniform 分量不可调**。打破它的 variant 都在重加权这个分量：仓库 LM（带 WD）朝"旧内容衰减"（recency）方向；CPU probe 中不加 WD 的 unconstrained 反而把均值放大（composite mean gain 3.5–6.2，逐层看集中在前几层，即加重 embedding/早期内容，类似 AttnRes 的 embedding sink 与 value residual；depth-scaled init 下依然如此）。方向因任务/规模而异，杠杆是同一个。在全正交 transport 中这种重加权以**无损**方式实现：mean gain `a = cos θ < 1` 把主通道内容等距地"旋入" complement 而不是销毁，`r` 通道可以把它旋回，新写入落在哪个子空间由 write 向量决定。对 RMSNorm 读取而言，这等价于**对主通道做可调的 depth 重加权（例如抵消 Pre-LN dilution / curse of depth / "no-erase"，或保持 embedding 的显著性）而不丢信息，也不损失梯度**（全状态仍是等距）。token 依赖的交换角 `θ(x)` 让 uniform kernel 变成随路径累积交换量衰减的 `M_kj(x)`（例如相继交换平面互相正交时恰为 `∏ cos θ_i(x)`，小角度下 ≈ `exp(−Σ θ_i(x)²/2)`），即**选择性无损遗忘**——Mamba 式选择性衰减的 depth 版，但因为有额外 streams 做"回收站"而可以无损。换言之：**额外 streams 的价值不是被保存的记忆，而是让主通道能够无损遗忘的存档空间。** 这也与 AttnRes 学到的 depth kernel（近对角 recency + 持续的 embedding sink）、"additive identity path 在深层/大规模上决定性"（Delta AttnRes、Review Residuals）以及 convex Highway 超过 ~20 层失效等 2026 年经验规律一致：衰减必须是无损的、全状态必须是等距的。

### 3.5 次要原因

- baseline 不是 faithful mHC（静态、无 dynamic、`diag_bias` 固定为 4，且被 WD）。
- 预算：20M tokens，1 seed；无噪声地板。
- 叙事在错误的量上做文章（complement 能量），而真正的竞争（2026 年 1–9 月）在 dynamic read/write 与 mixing 结构上推进得更快。

---

## 4. Q3：当前 literature 是否已经吃掉主要 novelty

| 主张 | 状态 | 先行工作 |
|---|---|---|
| Birkhoff 1^⊥ 收缩（含显式速率） | 已发表 | Zwick C-mHC（2026-08, `‖∏H−J/n‖ ≤ √n∏(1−nδ)`，L=64 测到 4.2e-4）、oHC、Homogeneity Trap、JPmHC |
| 正交 HC（Cayley / Stiefel / quaternion / NS polar） | 已发表 | JPmHC、EΔ-MHC-Geo 2605.06729、oHC（3.9B，含 15 步 Schulz polar 版） |
| signed / 谱范数约束的保均值 HC | 已发表 | sHC 2603.20896（IsoHC 恰是其 `Σ=I` 极限） |
| fixed-vector 正交 HC（IsoHC 本身） | 未有人训练，但被 oHC 命名为 SO(n−1) 对照并预测 ≈ iHC | oHC |
| 双边奇异值界 / bi-Lipschitz | 实质已覆盖；可调带宽只是增量 | oHC、Zwick、JPmHC |
| HC = depth-wise SSM / m-semiseparable | 已发表 | Kimi AttnRes 2603.15031 §6；Residual Stream Duality 2603.16039 |
| delta rule / Householder along depth | 已发表 | DDL 2601.00417 |
| stream collapse、symmetry breaking | 已发表 | 2606.03483（LSS）、2609.05309 |
| **static gauge 定理及其推论** | **未发现** | 与状态空间实现理论（相似变换）同源，但无人对 HC 陈述，也无人推导 read/write 封闭性条件 |
| **HC 的 controllability / observability / Hankel 分析** | **未发现** | — |
| **训练过的 mean-preserving 对照臂** | **未发现** | oHC 明确留作 open |
| **exchange 的功能解释（无损选择性 recency）** | **未发现**（oHC 只说到"stream diversity"） | — |
| **HC 的 depth-μP / 完整参数化** | **未发现** | CompleteP 2505.01618 只覆盖单 stream |

大局：attention-over-depth（Kimi AttnRes 48B 及至少 7 篇后续）正在超过 HC-mixing 这条线；生产模型在"保留 mHC"（DeepSeek-V4、GLM-5.3）与"扔掉 H_res、加强 read/write"（Qwen3.8、Hy4）之间分化。"更好的 transport 流形"这个方向已经拥挤且边际收益在下降。

---

## 5. Q4：还有哪些真正具有 ICML/NeurIPS 级潜力的突破口

按期望价值排序：

**(A) Hyper-connections 的 gauge 理论 + isometric exchange 机制（推荐，见 §6）。**
统一解释 2026 年分散的经验记录：为什么 H_res 在富 read/write 下冗余（gauge）、为什么 iHC ≥ mHC（同 gauge 类中的 dissipation）、为什么 oHC > iHC（非 gauge 的交换/curvature）、为什么 IsoHC ≈ iHC（类内 gauge），并给出更便宜的最小参数化。

**(B) 多 stream residual 的 depth-μP / 完整参数化。**
固定温度的 Birkhoff 在深度极限下 complement 消失（composite → 秩 1）；需要 transport 偏离 identity 的幅度 ~1/L、read/write 耦合 Θ(1) 才有非退化极限；小初始化 + WD 形成乘积型鞍点，解释 stream collapse。与 (A) 共享数学，可作为 (A) 的第二部分或独立后续。

**(C) 权重共享 / looped depth 中的 holonomy（高风险 side bet）。**
这是唯一一个静态 transport **不是** gauge 的场景：tied 的 H 在循环 T 次后留下 holonomy `H^T`，其谱是物理的；Birkhoff 会按 `ρ^T` 抹掉循环携带的 registers，而测试时增加迭代次数正是 latent reasoning / test-time compute 的热点。风险：identity 同样不收缩，等距旋转只是多了一个"迭代时钟"，未必更好。

**(D) "是否需要 expanded state" 的归因研究（对照 AttnRes / Block AttnRes）。** 更像 benchmark，信息价值高但 novelty 与算力要求不匹配。

不推荐：IsoHC-v2（depth-budgeted bi-Lipschitz、`Q·exp(S)`）——相对 oHC/Zwick/sHC 是增量，且静态版本仍是 gauge。

---

## 6. Q5：最值得继续追的一条路线

**工作标题：** *Residual mixing is (mostly) a gauge: what hyper-connections actually learn*

**核心主张（全部可证伪）：**
1. 【理论】静态 transport 是纯 gauge（定理 + read/write 封闭性条件）；gauge 不变量是 depth kernel `M_kj(x)` 及其 Hankel 谱。mixing 的物理内容只有：token 依赖（curvature）、相对 read/write 约束的破缺（mean↔complement exchange）、tied 循环的 holonomy；dissipation 只决定可达性/条件数。
2. 【理论】保均值 + 均值归一的 read/write ⇒ uniform kernel 钉死为全 1；全正交 transport 解除这一约束，且等距保证 read/write 范数不膨胀、梯度不衰减。
3. 【实证，决定性】在 faithful dynamic 框架中只改 H_res：dyn-IsoHC ≈ iHC；exchange-only ≈ oHC > iHC；lossy leak < exchange（尤其在深层）。
4. 【机制】exchange 降低主通道范数随深度的增长、提高后半层的有效贡献（curse of depth 指标），交换角在特定 token/位置上集中。
5. 【实用】最小 **gauge-fixed exchange HC**：identity transport + dynamic read/write + 每层 1–(n−1) 个 token 依赖的 Givens 交换角（mean 与 1^⊥ 方向之间）+ 可学习 exit readout；任意 n，无 Sinkhorn/polar/quaternion，比 oHC 更便宜。

**为什么是这条：**
- 它是唯一一条把仓库现有资产**全部变成贡献**的路线：mean/complement 分解是理论坐标系；IsoHC 实现就是 oHC 缺失的对照臂；WD 发现与 gauge 定理解释了自己的 null result。
- 它解释的是 2026 年真实存在的困惑（H_res 到底有没有用、为什么 oHC 赢、为什么 production 在分化），而不是在拥挤的"更好流形"赛道里再加一个流形。
- 决定性实验在 ~100M 规模即可做，kill 条件清晰。

**主要风险：**
- 审稿人可能认为 gauge 部分是"实现理论教科书内容"——贡献必须落在推论、预测与最小参数化的实证上。
- oHC 作者已把"训练 SO(n−1) 臂 + 可学习 readout"列为下一步；时间窗口约 3–4 个月（ICML 2027 截稿约 2027 年 1 月底）。
- oHC 的效应（0.019 BPB @ 3.9B/73B tokens）在 100M 规模可能低于噪声；需要用更深的模型放大深度效应，并依赖信噪比更高的机制指标。
- **CPU probe 的警示（附录 B）：** 在 0.8M 参数的小模型上，等距交换相对 identity 没有可检测收益；唯一稳健的杠杆是非等距的、前几层放大的主通道增益，与已知的可学习 residual 权重（LAuReL-RW，已进 Gemma 3n）重合。我对 Stage 1 通过的主观概率约 25–35%；最可能的失败形态是"缩减性"结论，即 HC 的小规模收益 ≈ 可学习主通道增益 + 富 read/write。

---

## 7. Q6：下一批信息增益最高的实验 / 理论工作

**Stage 0（本周，零 GPU）**
- 写清三条定理及证明：静态 gauge + 封闭性条件；保均值 ⇒ uniform kernel 钉死；等距交换 ⇒ 距离相关重加权 + 全状态梯度保持。
- 实现 gauge 不变诊断替换现有诊断：depth kernel `M_kj`（token 平均与方差）、其 Hankel 奇异值（有效 stream 数）、uniform-kernel 衰减曲线、主通道范数随深度增长、交换角统计。
- 修复训练配置：mixing / gate / read-write 参数不加 WD，并记录其轨迹。

**Stage 1（3–4 周，一张 5090；决定性筛选）**
faithful dynamic 框架（mHC 式 σ-read / 2σ-write，系数由 RMSNorm(flatten X) 线性生成），只改 H_res；**所有臂统一使用 depth-scaled init（输出投影 std 0.02/√(2L)）**，mixing / gate 参数不加 WD：
1. baseline（单 stream）
1b. baseline + 每层可学习 residual 标量（LAuReL-RW 式 `x_{k+1} = g_k x_k + f(x_k)`）——强对照（见附录 B：CPU probe 中最大的收益来自非等距的、前几层放大的主通道增益，depth-scaled init 不能消除它，所以必须用可学习 residual 权重来对照）
2. iHC（H_res = I）
3. mHC（Sinkhorn，mixing 参数无 WD）
4. dyn-IsoHC（SO(n−1)，保均值）——oHC 缺的对照
5. dyn-SO(n)（oHC 型；n=4 可用 quaternion 或 NS polar）
6. dyn-exchange-only（只有 mean↔1^⊥ 的 Givens 角，任意 n）
7. dyn-lossy-leak（token 依赖的主通道衰减 g≤1，无存档，非等距）
7b. dyn-free-gain（token 依赖的主通道增益 g∈(0,2)，非等距）——区分"等距交换"与"任意均值增益"
8. (5) 或 (6) + 可学习 exit readout

规模：~100–150M、24–48 层、2–3B tokens；全部臂 1 seed，(2)(4)(5)(6)(7) 第二 seed 估噪声地板。指标：val loss、若干 BPB 下游、§Stage 0 的 gauge 不变诊断、后半层 layer-skip 敏感度（Csordás 式）。

**Stage 2（通过后）**
- 深度扫描（24→96 层 deep-thin，同参数量），预测 exchange 相对 iHC 的优势随深度增大、lossy leak 随深度变差。
- 350M 确认；与 Block AttnRes / Delta AttnRes 对比。
- 可选：looped / tied-HC 的 holonomy 实验（§5(C)）。
- 写作目标：ICML 2027；备选 NeurIPS 2027。

---

## 8. Q7：什么结果出现就继续，什么结果出现就终止

**继续（全部满足）：**
- dyn-IsoHC 与 iHC 的差 < 1σ（解耦预测成立；动态情形不受静态定理覆盖，所以这是真正的检验）；
- dyn-SO(n) 与 exchange-only 都比 iHC 好 ≥ 2σ，且 exchange-only 恢复 oHC 增益的 ≥ 70%；
- lossy leak 劣于 exchange（至少在最深配置上 ≥ 2σ），且 exchange 相对 iHC 的收益在 1b（可学习 residual 标量）存在时仍然成立；
- 至少一个机制指标（主通道范数增长变慢 / 后半层贡献上升）出现且方向一致。

**终止：**
- 在 100M+ / 2B+ tokens 上 dyn-SO(n) 不能以 ≥ 2σ 胜过 iHC ⇒ 在我们可及的规模上复现不了核心现象，HC-mixing 不是我们可用的杠杆，停止整个方向；
- dyn-IsoHC ≈ oHC ⇒ exchange 假设错误，且方法等于 oHC，停止；
- lossy leak ≈ exchange，或 1b（单 stream + 可学习 residual 标量）≈ exchange ⇒ 存档没被用，收益只是普通的 residual 重加权（LAuReL / LayerNorm Scaling / depth-scaled init 已占），停止或只保留理论短文。

**预算上限：** Stage 1 约 300–500 GPU-hours。若到期仍无 go 信号，写短文（gauge 定理 + WD artifact + 训练过的 SO(n−1) 对照）后停止。

---

## 9. 对第一性原理问题的简答

- **为什么需要多个 streams？** 单条加性 residual 有三个缺陷：不能无损遗忘（只能加，不能擦；norm 增长、dilution、curse of depth）；uniform depth kernel 不可调；按层路由信息必须占用特征子空间。多 streams 提供一个等距的"侧存储"，使得 (1)(2) 可以经由 exchange 完成，(3) 可以经由 complement 的 read/write 完成。小规模 CPU probe 里 (2) 表现为前几层放大（embedding anchoring），且只有非等距增益能做到。生产证据显示真正起作用的是富 dynamic read/write 与 exchange，而不是保均值的 mixing。
- **mHC 的 contraction 是 pathology、regularization 还是无关？** 在仓库里无关（解耦 + WD 决定）。在生产规模上是**容量浪费型病理**而非稳定性病理：主通道从不收缩所以训练稳定，但额外 streams 在几层内被同质化（oHC 测到 24 个 sublayer 后 σ_min 7.8e-7；mHC vs baseline 在 3.9B 上不显著）。没有证据表明它是有益正则（iHC ≥ mHC）。更大的问题不是收缩，而是保均值禁止了 exchange。
- **旧 IsoHC hypothesis 中真正有价值的部分？** mean/complement 分解作为坐标系；"Birkhoff 会浪费额外状态"这一方向性判断（已被他人证实）；以及 IsoHC 本身作为 exchange 假设的对照。
- **残差流真正需要保留什么？** 在 Pre-RMSNorm 读取下，流的欧氏能量对前向无关（读取尺度不变）。需要的是：全状态 carry 的良好条件（梯度传输）；read-visible 子空间的组成可控（recency / dilution）；信息对未来 read 的可达性（observability，由 read/write 参数化决定，不由 transport 决定）。
- **更合理的 residual transport objective：** 全状态等距（或偏离 ~1/L）的 carry + token 依赖的 read-visible ↔ 存档子空间交换 + 足够富的 read/write。只在 1^⊥ 上等距且固定均值，是这个目标里错误的特例：它在没人读的子空间里保能量，同时禁止了唯一能让它有用的操作。

---

## 附录 A：可复现脚本

所有脚本在仓库根目录运行，只需 CPU + PyTorch：

```bash
python3 docs/0924_salvage/scripts/wd_attribution.py   # §3.2
python3 docs/0924_salvage/scripts/gauge_check.py      # IsoHC → identity-HC 逐点等价
python3 docs/0924_salvage/scripts/gauge_table.py      # §3.3 表
# CPU probe（需先下载语料到 scripts/tinyshakespeare.txt）
curl -o docs/0924_salvage/scripts/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
cd docs/0924_salvage/scripts && ./run_sweep.sh && ./run_sweep2.sh && ./run_sweep3.sh && ./run_sweep4.sh   # 4 = depth-scaled init
python3 summarize.py ../cpu_probe_runs
```

## 附录 B：CPU accessibility probe

**设置：** 仓库的 `TwoBranchHCTransformer`（静态 HC，n=4），16 层（32 个 transport），d=64，4 heads，ctx 64，char-level TinyShakespeare，batch 32，1200 步，AdamW lr 3e-3；HC 专属参数（mixing、λ、read/write 向量、stream_embed）**不加 WD**，其余 ≥2D 参数 WD 0.1。额外臂（仅在 probe 内定义，仓库未改）：`orthogonal` = 仓库已有的 `OrthogonalMixing`（全 O(n)，不固定 1）；`exchange` = mean 方向与一个固定 1^⊥ 方向之间的可学习 Givens 旋转（等距、打破保均值）；`leaky` = 可学习标量均值增益 `g=2σ(s)∈(0,2)`、complement 恒等（非等距，相当于 HC 形式的 LAuReL-RW）。`clamp` = 在所有状态把 stream 替换为其均值后的 Δloss（complement 的因果使用量）；`mGain` = 32 个 transport 复合后的均值增益 `1ᵀΦ1/n`。

**λ 初值 0.01（训练中自由增长），两个 seed：**

| arm | val loss (seed 0 / 1) | 均值 | clamp Δ | composite mGain |
|---|---|---:|---|---|
| unconstrained | 1.8109 / 1.8235 | **1.817** | 0.011 / 0.039 | 4.35 / 6.07 |
| leaky（自由标量均值增益） | 1.8306 / 1.8499 | **1.840** | 0.004 / 0.002 | 2.89 / 2.05 |
| orthogonal（O(n)） | 1.8545 / 1.8736 | 1.864 | 0.210 / 0.090 | 0.27 / 0.31 |
| exchange（Givens） | 1.8609 / 1.8753 | 1.868 | 0.078 / 0.023 | 0.59 / 0.77 |
| identity-HC | 1.8542 / 1.8854 | 1.870 | 0.002 / 0.004 | 1 |
| IsoHC | 1.8734 / 1.8793 | 1.876 | 0.015 / 0.022 | 1 |
| baseline（单 stream） | 1.8847 / 1.8689 | 1.877 | — | — |
| mHC（diag_bias 4） | 1.8778 / 1.8833 | 1.881 | 0.009 / 0.005 | 1 |

**λ 初值 0.5（强耦合），seed 0：** unconstrained 1.8054、leaky 1.8300、exchange 1.8361、orthogonal 1.8396、identity 1.8433、IsoHC 1.8519、mHC 1.8749（diag_bias 2 为 1.8751）。complement 被大量使用（clamp Δ：orthogonal 1.19、IsoHC 0.55、exchange 0.54、mHC 0.29、identity 0.21），但 loss 与弱耦合时相比只好 ~0.01。

**depth-scaled init 对照（输出投影 std 0.02/√(2L)，seed 0）：** baseline 1.8847（与未缩放几乎相同）、identity 1.8849、IsoHC 1.8653、exchange 1.8794、leaky 1.8478、unconstrained 1.8160；两种自由增益臂仍然学到同样的早层放大剖面。

**解读（只作方向性证据；这是 0.8M 参数的 char 模型）：**
1. 同配置不同 seed 的差异可达 0.03（identity 1.854 vs 1.885），所以 identity / IsoHC / orthogonal / exchange / baseline / mHC 之间的差别都在噪声内。**IsoHC 相对 identity 没有可检测的收益；等距交换在这个规模上也没有。**
2. 唯一稳健的杠杆是**非等距、按深度剖面化的主通道增益**：unconstrained 比 identity 好 ~0.05（≈3σ），leaky 好 ~0.03（≈2σ），且在 scaled init 下依然成立。逐层日志显示两者都学到**前几层强放大、之后 ≈1 或略小于 1**（leaky seed 1：1.26, 1.22, 1.17, 1.16, 1.11, 1.10, 1.06, … → 0.93；unconstrained：2.06, 1.79, 1.48, 1.44, …），即让 embedding / 早层表示在主通道中占 ~3× 权重——与 AttnRes 的 embedding sink、value residual 的经验一致，也正是保均值 transport（mHC、IsoHC、iHC）在结构上禁止、而等距 transport（|mean gain| ≤ 1）无法通过均值增益实现的操作。
3. 训练不会推动 mHC 离开初始收缩（0.9306→0.929/0.932；0.615→0.60–0.61），即便 complement 被强耦合使用——与 §3.2 一致。
4. 对 §6 路线的含义：定理层面的主张（保均值钉死 uniform kernel、放开它是杠杆）得到支持；但"等距交换本身是杠杆"在小规模上**没有**证据，其证据目前只来自 oHC 的 3.9B/73B tokens。小规模上起作用的部分与 LAuReL-RW 式可学习 residual 权重高度重合——这就是 Stage 1 必须包含 1b、7b 对照的原因，也意味着 Stage 1 很可能给出"缩减性"结论（HC 的小规模收益 ≈ 可学习主通道增益 + 读写）。


## 附录 C：文献访问的局限

arXiv / OpenReview / Semantic Scholar / HuggingFace 在本环境被屏蔽；WebSearch 配额在调查末尾用尽。全文读过（经 GitHub 镜像或作者仓库）：oHC 2609.02672、xHC 2607.14530、Qwen3.8-Next 2608.30320、Kimi AttnRes 2603.15031、DDL 2601.00417、Zwick C-mHC（GitHub，arXiv 状态未知）、Residual Stream Duality、Multi-Head AttnRes、Delta AttnRes（作者博客）、DeepSeek-V4 / GLM-5.3 / Hy4 的 HF 代码。其余（JPmHC、sHC、EΔ-MHC-Geo、Homogeneity Trap、stream collapse 2606.03483、2609.05309 等）为摘要级，定量细节在引用前需再核实。
