# 理论基础：严格 adjoint 读写与残差状态

更新：2026-09-26。本文是后续研究的数学基线，替代旧稿中从“补空间能量被保护”直接推到“任务更好”的推断。旧稿与实验记录保留为历史材料。本文区分**已证明的代数结论**、**依赖参数类的结论**与**尚待实验检验的假说**；证明正确不等于新颖性成立，也不等于有任务收益。

当前方案（2026-09-26，R4）为：**两流严格 adjoint 读写、signed carrier 初始化与可学习的连续动态路由**，见 §10；完整模型、合同核验和合成NTP集成训练已完成，尚无自然语言质量收益证据。§1–8 保留先前 HC 审计，不能把其中的旧参数类或实现描述自动套到 R4。§9 的保留—复用分析和受保护槽位合同服务过 RDM 提案；RDM 已撤销主线资格，这些局部结论不恢复其研究优先级。当前方法与实验状态见 [研究入口](README.md) 与 [架构选择](architecture.md)。

## 1. 历史HC实现与审计对象（R4见§10）

将每个 attention/MLP 分支视为一个深度步骤，记步骤数为 \(N\)，流数为 \(n\ge2\)。省略 batch、token 维，将它们纳入每个流的特征维度。被审计的历史HC模型为

\[
z_k=c_k^\top X_k,\qquad c_k=a_k/n,\qquad
\delta_k=f_k(z_k),\qquad
X_{k+1}=H_kX_k+b_k\delta_k.
\tag{1}
\]

最终读出为 \(z_N=c_N^\top X_N\)，之后接最终归一化与词表投影。\(X_k\in\mathbb R^{n\times d}\)，\(a_k,b_k\in\mathbb R^n\)；\(f_k\) 可以是非线性的 attention 或 MLP，不要求线性化。

源码对应：

| 对象 | 当前实现 | 理论边界 |
|---|---|---|
| 单 block HC | [lm/models.py](../../lm/models.py) 的 `HCTransformer` | 一个 block 合并 attention/MLP 更新；读写向量的底层权重初始化为零 |
| 两分支 HC | [lm/models.py](../../lm/models.py) 的 `TwoBranchHCTransformer` | 每层两个 transport；读写底层权重初始化为随机零均值向量 |
| transport | [lm/mixing.py](../../lm/mixing.py) | `forward()` 不接收状态，所有这些 mixer 都是静态、逐层独立的参数 |
| 读写 | `_make_readout`、`_make_stream_vector` | \(a=\mathbf1+\lambda_aP_\perp w\)，\(b=\mathbf1+\lambda_bP_\perp v\)；\(\lambda\) 可学习且跨层共享，读写可有负分量 |
| 入口 | `stream_embed` | \(X_0=\mathbf1x_0+S_0\)；两分支初始化将 \(S_0\) 中心化，但它仍是可训练参数 |
| 投影 | [isohc/projection.py](../../isohc/projection.py) | 理论是精确正交；有限步 NS、基础矩阵精度和回退使实际结果近似成立 |
| transport 诊断 | [lm/transport_analysis.py](../../lm/transport_analysis.py) | 本轮 schema 2 修复完整复合；旧 schema 1 只适用于共同保持补空间的 mixer，详见 §7.2 |

原始 [HC](https://arxiv.org/abs/2409.19606) 与 [mHC](https://arxiv.org/abs/2512.24880) 是架构参照，当前仓库静态实现是受控代理，不能自动视作生产动态 mHC 的复现。

## 2. 保留的几何事实，以及它们不能推出的结论

令

\[
e_0=\mathbf1/\sqrt n,\quad P_0=e_0e_0^\top,\quad
P_\perp=I-P_0,\quad U^\top U=I,\quad U^\top e_0=0.
\]

在正交基 \(E=[e_0,U]\) 下，一般 transport 写成

\[
E^\top HE=\begin{pmatrix}g&r^\top\\q&B\end{pmatrix}.
\tag{2}
\]

\(g\) 是带符号的均值到均值系数，\(q,r\) 是均值与差异子空间之间的交换，\(B\) 是补空间压缩。\(\|H\mathbf1-\mathbf1\|\) 只能衡量偏离，不能判定是放大、衰减还是旋转，也不能给出 \(g\) 的符号。

**命题 1：双向均值保持与块对角结构等价。**

\[
H\mathbf1=\mathbf1,\quad \mathbf1^\top H=\mathbf1^\top
\iff E^\top HE=1\oplus B.
\tag{3}
\]

证明：右固定向量使第一列为 \((1,0)^\top\)，左固定向量使第一行为 \((1,0)\)；反向直接乘回。仅有一个约束不够使两个交换块同时为零。

在 (3) 下，记 \(\mu_k=\mathbf1^\top X_k/n\)、\(Y_k=U^\top X_k\)、\(\alpha_k=U^\top a_k\)、\(\beta_k=U^\top b_k\)，并要求 \(\mathbf1^\top a_k=\mathbf1^\top b_k=n\)，则

\[
\begin{aligned}
z_k&=\mu_k+\alpha_k^\top Y_k/n,\\
\mu_{k+1}&=\mu_k+\delta_k,\\
Y_{k+1}&=B_kY_k+\beta_k\delta_k.
\end{aligned}
\tag{4}
\]

因此保均值 transport 禁止了 **transport 内部**的直接交换，却没有关闭经由非线性分支的读写：\(Y_k\to z_k\to\delta_k\to\mu_{k+1},Y_{k+1}\)。把 (3) 解读为“额外流无效”是错误的。

**命题 2：Birkhoff 仅保证非扩张；IsoHC 保证 transport 等距。**

若 \(H\) 非负且双随机，由其为置换矩阵的凸组合可得 \(\|H\|_2\le1\)，从而 \(\|B\|_2\le1\)。它可能收缩，也可能是置换而完全不收缩。例如

\[
H_\rho=P_0+\rho P_\perp,\quad 0\le\rho<1
\quad\Longrightarrow\quad B^N=\rho^N I.
\tag{5}
\]

若 \(Q^\top Q=I\) 且 \(Qe_0=e_0\)，则 \(Q^\top e_0=e_0\)，故 \(Q=P_0+URU^\top\)、\(R\in O(n-1)\)，且

\[
\mu(QX)=\mu(X),\qquad \|P_\perp QX\|_F=\|P_\perp X\|_F.
\tag{6}
\]

证明已经包含在块分解与正交性中。这些是正确的几何基础；相关方向已有广泛先行工作，包括 [oHC 的均值/差异分析](https://arxiv.org/html/2609.02672v1)。它们不证明 LM 质量改善，不证明整个网络的梯度等距，也不证明某条被保留的方向是可读取或任务相关的。

即使 \(H_k\) 精确正交，在固定静态读写下整个步骤的 Jacobian 仍是

\[
J_k=H_k\otimes I_d+(b_kc_k^\top)\otimes Df_k(z_k).
\tag{7}
\]

第二项可导致放大或消失。对 attention，可将整段 token 特征展平后理解 \(d\)；输入依赖的读写/transport 还会增加路由导数项。

## 3. 静态 gauge 定理：必须连同读写与边界一起变换

### 定理 1：可逆静态 transport 的代数消去

假设每个 \(H_k\) 静态且可逆。取任意可逆 \(G_0\)，递归定义

\[
G_{k+1}=H_kG_k,\qquad X_k=G_kZ_k.
\]

同时变换

\[
\tilde c_k=G_k^\top c_k,\qquad
\tilde b_k=G_{k+1}^{-1}b_k,\qquad
Z_0=G_0^{-1}X_0,\qquad
\tilde c_N=G_N^\top c_N.
\tag{8}
\]

则原网络逐步等价于

\[
Z_{k+1}=Z_k+\tilde b_k f_k(\tilde c_k^\top Z_k),
\qquad z_N=\tilde c_N^\top Z_N.
\tag{9}
\]

**证明。** 代入 \(X_k=G_kZ_k\)，左乘 \(G_{k+1}^{-1}\)，由 \(G_{k+1}^{-1}H_kG_k=I\) 得 (9)。由 (8) 得每一步 \(\tilde c_k^\top Z_k=c_k^\top X_k\)，故即使 \(f_k\) 非线性，分支输入与输出仍逐步相等；最终读出、归一化与 logits 同样相等。训练态 dropout 需耦合同一随机掩码，或仅作输出分布相等的陈述。证毕。

该定理证明的是**给定参数点的函数重表示**。要声称“两个架构具有相同函数类”，必须另外验证 (8) 中的入口、最终出口、读写向量、共享关系以及约束均落在比较架构的允许参数类中。取 \(G_0=I\) 可保持入口不变，但出口与逐层读写仍必须变换。

### 推论 1：当前静态 IsoHC 与 identity-HC 的函数类相同

对当前参数化，\(a,b\in\mathcal A=\{v:\mathbf1^\top v=n\}\)。若每个 \(H_k\in O(n)\) 且固定 \(\mathbf1\)，则 \(G_k\) 也属于这一子群，且

\[
\mathbf1^\top G_k^\top a_k=n,\quad
\mathbf1^\top G_{k+1}^{-1}b_k=n.
\]

因此变换后的读写仍在 \(\mathcal A\)。共享 \(\lambda_a,\lambda_b\) 不阻碍这一结论：底层中心化权重分别变换为 \(G_k^\top P_\perp w_k\) 与 \(G_{k+1}^{-1}P_\perp v_k\)，可以保留同一个 \(\lambda\)。若 \(\lambda=0\)，对应向量等于 \(\mathbf1\)，变换后仍不变。取 \(G_0=I\) 还保留当前入口与 token/词表权重绑定。

正交变换保持这些**有效读写向量及中心化底层权重的欧氏范数**。反向 identity 是 IsoHC 的成员，故在精确算术与上述参数类中，两者函数类相同。

这不等于两者训练过程相同：原始 mixer 参数、投影坐标、初始化分布、Adam 的逐坐标二阶状态和参数组 weight decay 均可能破坏优化过程的等价性。当前实现还存在有限精度误差。“仅可能改变优化、正则化和计算代价”是此处的正确含义；不能据此推出“训练收益必为零”。

### 推论 2：不保均值的正交 mixer 也不是普遍的表达力例外

在入口、出口、读写均自由的类中，任意静态 \(Q_k\in O(n)\) 同样适用定理 1，且读写欧氏范数不变。若要求 \(\mathbf1^\top a=\mathbf1^\top b=n\)，变换后的向量通常不再满足约束，于是**在此受限类中**不能直接由定理判定等价。

因此，均值交换是否扩大可实现路由，取决于相对于哪种读写/边界类作比较。“交换一定有表达力，而 IsoHC 一定没有”不是脱离参数类即可成立的命题。

### 推论 3：可逆 mHC 的代数等价不等于无代价等价

双随机、可逆的静态 \(H_k\) 及其乘积保留两侧固定向量，故 (8) 仍保持当前仿射读写类。但非正交变换通常不保持范数，且

\[
\|\tilde b_k\|\le\|G_{k+1}^{-1}\|_2\,\|b_k\|.
\tag{10}
\]

这是一个最坏情况上界，不是“实际 write 必然放大”的等式。若 \(b_k=\mathbf1\)，那么 \(G^{-1}b_k=\mathbf1\)，即使补空间极度收缩也不放大。是否付出巨大范数成本，取决于 write 在弱奇异方向上的投影。用复合谱中的平均值直接宣称所有 write 必须放大相同倍数不成立。

正元素的双随机矩阵也可能奇异，如 \(P_0\)。奇异 transport 不满足定理 1；有限步 Sinkhorn 还可能有行列和误差，必须报告误差后再应用近似版本。[现有结果](evidence.md) 的 48L/24L mHC 行和误差均值约为 `0.00463`/`0.00614`，不是可直接忽略的舍入误差：当前 raw 实现不严格满足本推论的双向固定向量条件，不能宣称其在固定和读写类内精确 gauge 等价。只要矩阵可逆，在完全自由读写类中的定理 1 仍然适用。不能把“任意静态 transport 都是 gauge”用作不加条件的定理。

**删除干预与 gauge 变换不同。** 仅令 \(H_k\leftarrow I\) 而保持原读写/出口不变，一般会改变函数。其 loss 上升不反驳定理；其 loss 不变也不证明重训后的两类必然等价。

## 4. 有效 depth kernel：transport 坐标之外的对象

定义有向乘积

\[
\Phi(t,s)=H_{t-1}\cdots H_s\quad(t>s),\qquad \Phi(s,s)=I.
\]

直接展开 (1) 得

\[
z_t=c_t^\top\Phi(t,0)X_0+
\sum_{j=0}^{t-1}K_{tj}\delta_j,\qquad
K_{tj}=c_t^\top\Phi(t,j+1)b_j.
\tag{11}
\]

这里 \(t=N\) 可以是最终出口，\(j<t\) 保证先写后读。\(K\) 是固定分支输出坐标下的**开环深度路由核**；它不意味着 \(\delta_j=f_j(z_j)\) 是外生或相互独立的真实数据，也不意味着整个网络是线性的。[Attention Residuals §6.2](https://arxiv.org/abs/2603.15031) 已明确给出 HC 的深度核乘积、半可分结构和状态扩展解释；因此 rank 观点本身也不应作为本项目创新。这些量用于审计新工作流实际保留和读取了什么，不替代 §9 的方法目标与任务验证。

**命题 3：\(K\) 与入口作用在可逆流坐标变换下不变。**

对任意逐层可逆 \(G_k\)，设

\[
\tilde H_k=G_{k+1}^{-1}H_kG_k,\quad
\tilde b_k=G_{k+1}^{-1}b_k,\quad
\tilde c_k=G_k^\top c_k.
\]

由中间因子抵消，

\[
\tilde\Phi(t,s)=G_t^{-1}\Phi(t,s)G_s,
\quad
\tilde c_t^\top\tilde\Phi(t,j+1)\tilde b_j=K_{tj}.
\tag{12}
\]

入口项在同时变换 \(X_0\) 后也相等。此结论不要求 \(H_k\) 可逆；仅在进一步把 \(H\) 全部化为 identity 时才需要可逆性。

不变性范围必须明确：如果把分支输出另行缩放 \(\delta_j\mapsto s_j\delta_j\)、write 逆向缩放，\(K\) 的列会变化。故跨模型比较 kernel 谱时须固定分支输出的单位、归一化和读出定义，或报告数据加权的预测误差；“gauge 不变”不是对所有神经网络重参数化均不变。

### 定理 2：跨深度切分的可达可观测阶不超过流数

在切分 \(c\) 处，过去的分支索引为 \(j<c\)，未来读出为 \(t\ge c\)。定义

\[
R_c=\big[\Phi(c,j+1)b_j\big]_{j<c}\in\mathbb R^{n\times c},
\qquad
O_c=\big[c_t^\top\Phi(t,c)\big]_{t\ge c}.
\]

跨切分矩阵（有限时变情形的 Hankel 型矩阵）为

\[
\mathcal H_c=[K_{tj}]_{t\ge c,j<c}=O_cR_c,
\qquad r_c=\operatorname{rank}\mathcal H_c\le n.
\tag{13}
\]

**证明。** 由 \(\Phi(t,j+1)=\Phi(t,c)\Phi(c,j+1)\) 得分解；秩上界为矩阵乘积秩不超过中间维度。进一步，由线性映射限制在 \(\operatorname{im}R_c\) 上的秩—零度定理，

\[
r_c=\dim\operatorname{im}R_c-
\dim\big(\operatorname{im}R_c\cap\ker O_c\big).
\tag{14}
\]

即只有过去能够写入、未来能够读出的部分计入 \(r_c\)。证毕。

若一个保持相同分支源与读出定义的 \(q\) 维线性路由状态实现同一个开环核，则每个切分都有 \(q\ge r_c\)。反之，单个切分的映射可由 rank 分解用 \(r_c\) 维状态传递。这一逐切分事实不自动给出满足所有参数共享、正性、光滑性和动态路由约束的全局实现。最小实现、可达/可观测与 Hankel 思想来自经典系统理论，参见 [Ho–Kalman 原始论文档案](https://ntrs.nasa.gov/citations/19670049337)；它们本身不应被包装为本项目发明。

### 决定性反例：identity-HC 可以拥有大于 1 的有效阶

取 \(n=2\)、单位向量 \(u\perp\mathbf1\)、所有 \(H_k=I\)，并令

\[
a_t=\mathbf1+\sqrt n\,\alpha_tu,\qquad
b_j=\mathbf1+\sqrt n\,\beta_ju.
\]

所有读写都满足 \(\mathbf1^\top a_t=\mathbf1^\top b_j=n\)，且

\[
K_{tj}=a_t^\top b_j/n=1+\alpha_t\beta_j.
\]

取切分前两个写入的 \(\beta=(0,1)\)，切分后两个读出的 \(\alpha=(0,1)\)，则

\[
\mathcal H_c=\begin{pmatrix}1&1\\1&2\end{pmatrix},
\qquad \det\mathcal H_c=1,\quad r_c=2.
\tag{15}
\]

这里没有 transport 均值交换、没有 token 依赖、没有 tied loop，仍有两个可达可观测深度状态。单流、任意逐层标量 residual gain 的同源路由，跨切分秩至多 1，不能实现 (15)。在任意 \(n\) 下，使用 \(n-1\) 个正交补方向并取仿射独立的读写，可以类似构造 \(r_c=n\)。

这严格反驳了“保均值约束本身让额外流失效”，并说明 token 依赖、均值交换、tied holonomy 三种 transport 机制不能穷尽**额外状态**的价值。反例并不声称 identity transport 本身不可消去；关于何时 \(H\) 在同类内不可消去，还要检查 §6 的边界与参数约束，三种机制的穷尽性并未被证明。**消去 \(H\) 不会消去多流状态和读写形成的多维深度核。**

反例的能力边界同样重要：它排除的是保留同一分支源的标量路由实现，不能据此宣称同宽或异宽、经过重新训练的任意单流非线性网络都无法完成同一任务。若用该反例支持额外状态的任务价值，需结合冻结主干干预和充分重训的单流强对照；核压缩只是可选的机制诊断。

分析实际状态时还要纳入入口。若允许的输入映射为 \(X_0=B_{\mathrm{in}}x_0+S_0\)，应将 \(R_c\) 扩充为 \([\Phi(c,0)B_{\mathrm{in}},R_c]\)，并单独核查固定偏置 \(S_0\) 的输出作用。当前共享 token/position 入口的流方向是 \(B_{\mathrm{in}}=\mathbf1\)。只分析分支间 \(K\)，可能漏掉早层均值增益对 embedding 路径的加权。增强后的跨切分核仍有秩上界 \(n\)，但不得把实际上不允许的独立入口扰动算作任务可达信号。

### 谱、数据相关性与近似阶

令 \(W_c=R_cR_c^\top\)、\(W_o=O_c^\top O_c\)。在流坐标变换下二者分别作逆合同与合同变换，\(W_cW_o\) 作相似变换；其非零特征值等于 \(\mathcal H_c\) 的非零奇异值平方。因此有限切分谱比单独状态能量更接近可输入—可读出的对象。

但奇异值大并不自动意味着任务重要：真实 \(\delta_j\) 可能强相关，或相关方向不影响 loss。可定义固定阈值下的谱阶、rank-\(q\) 核的最佳重构误差，以及在持出分支输出上的读出误差；随后必须让状态压缩在闭环推理中接受任务检验。不同 \(c\) 的小谱尾也不保证存在同一个低成本、约束合法的全网压缩。

## 5. 弱读写与 \(\lambda^2\)：精确式与条件

若每个 \(H_k\) 双向保均值，则 \(\Phi(t,j+1)=P_0+UC_{tj}U^\top\)，其中 \(C_{tj}=B_{t-1}\cdots B_{j+1}\)。记读写尺度为 \(\lambda_{a,t},\lambda_{b,j}\)：当前内部分支分别共享尺度，但最终出口拥有独立的 `readout_final_lambda`。将当前读写参数化代入 (11)，有精确恒等式

\[
K_{tj}=1+
\frac{\lambda_{a,t}\lambda_{b,j}}{n}
(P_\perp w_t)^\top\Phi(t,j+1)(P_\perp v_j).
\tag{16}
\]

从而

\[
|K_{tj}-1|\le
\frac{|\lambda_{a,t}\lambda_{b,j}|}{n}
\|P_\perp w_t\|\,\|C_{tj}\|_2\,\|P_\perp v_j\|.
\tag{17}
\]

只有当两个 \(\lambda\) 同阶且底层权重范数、复合增益及讨论的深度范围受到控制时，才可简写为 \(O(\lambda^2)\)。当前 \(\lambda\) 是可学习参数，\(w,v\) 也可增长；\(\lambda=0.01\) 的**初始化**不能单独证明训练后扰动永远为 \(10^{-4}\)。常数还依赖流数及权重归一化。

初始状态也是独立通道。若 \(X_0=\mathbf1x_0+S_0\) 且 \(\mathbf1^\top S_0=0\)，则

\[
c_t^\top\Phi(t,0)X_0
=x_0+\frac{\lambda_{a,t}}{n}(P_\perp w_t)^\top\Phi(t,0)S_0.
\tag{18}
\]

此项的量级为 \(O(\lambda_a\|S_0\|)\)，不是必然的 \(\lambda_a\lambda_b\)。流嵌入的学习或较大初始化会使其不可忽略。

此外，(16) 描述开环核系数，不是 loss 差异公式。许多小系数可随深度累积，非线性 Jacobian 可放大它们。因此“去掉 complement 的 loss 变化必为 \(10^{-4}\)”不成立。合法的定量结论应记录训练后的有效 \(\alpha,\beta\)、初始/中间 \(Y\)、核偏离、分支与最终 logits 灵敏度，再测干预。

两个有用的精确退化情形是：

1. 所有读出（含最终出口）都为 \(a_k=\mathbf1\)，transport 保持左均值且 \(\mathbf1^\top b_k=n\)：由 (4) 网络只读取均值，补空间不影响函数。
2. 初始补空间为零、所有 \(b_k=\mathbf1\)、transport 保持右均值：补空间一直为零，任意中心化读出都无信号可读。

第一种解释了单 block HC 在零读写底层权重初始化时的精确退化；两分支模型采用随机中心化权重，只有弱耦合而非相同的精确退化。不能将一种初始化的推论直接套到另一种。

## 6. 动态 transport 与 holonomy 的正确边界

### 6.1 动态不自动阻止代数消去

若 \(H_k\) 依赖输入、token 或当前状态，沿一条实现轨迹仍可递归设 \(G_{k+1}(x)=H_k(x)G_k(x)\)。当每一步可逆时，(8) 的代数抵消仍成立，但变换后的读写、入口/出口可能依赖完整路由历史，且原来简单的 router 未必能以相同参数量和计算从 \(Z_k\) 实现它们。

更强的障碍是，依赖状态的 \(X\mapsto G(X)^{-1}X\) 不保证构成全局一一可逆的状态坐标变换。故这里仅保证**沿轨迹的代数重写**，不能不经额外证明就声称有限架构的函数类或优化过程等价。

有明确反例说明“动态必然破 gauge”过强：若 \(G_k(x)\) 由某个外生输入条件直接可计算、所用读写类对这些变换封闭，则设 \(H_k(x)=G_{k+1}(x)G_k(x)^{-1}\)，这组动态 transport 仍可在同一允许类内消去。相反，实际有限宽、受正性或共享约束的 router 可能不封闭，这是需要证明和实验的内容。

对实现轨迹上冻结的 \(H,a,b\)，逐 token 仍可构造 (11) 与 (13)。但真实扰动会改变 router；把多个 token 核平均也可能增加秩。**冻结路径的标量跨切分秩上界 \(n\)，不是整个动态非线性网络闭环 Jacobian 的秩上界。** 真实闭环还包含特征、序列位置、分支导数与路由导数，需用相应的完整状态维度分析。

### 6.2 Looped depth 是一种约束，不是唯一例外

若要求入口与出口使用同一个坐标帧 \(G_N=G_0\)，闭环乘积

\[
\Omega=H_{N-1}\cdots H_0
\]

变为 \(\tilde\Omega=G_0^{-1}\Omega G_0\)，故其共轭类不变。若所有变换后的 \(\tilde H_k=I\)，必须有 \(\Omega=I\)。这才是这里使用 holonomy 的严格条件。

普通开放、逐层独立的链没有上述周期约束，最终出口可随 \(G_N\) 变换。tied loop 常要求同一 \(H,a,b,f\) 重复使用，层依赖的 gauge 通常破坏这种共享，于是等价类缩小；但某些可交换、特定读写或特殊周期结构仍可退化。不能写成“凡 looped 必然有不可消去 holonomy”。

其他限制也可能阻止定理 1 在同类内实现：固定出口、非负/有界读写、逐流归一化或非线性、稀疏支持、共享/量化参数、范数预算，以及奇异 transport。故 tied loop 绝不是“静态 transport 非 gauge 的唯一场景”。

## 7. 对旧诊断、head mixing 与投影的直接含义

### 7.1 能量的适用范围

一般 \(GL(n)\) 的坐标变换会改变 \(\|P_\perp X\|\)，并且可能改变何者被称为均值方向。在**固定均值的正交子群**内，补空间能量恰好不变；不能把它在所有 gauge 下都称为坐标伪影。正确区分是：

| 指标 | 能说明什么 | 不能单独说明什么 |
|---|---|---|
| 补空间能量 | 指定度量和均值方向下的状态量；在 IsoHC 子群内不变 | 可读取性、任务相关性、相对 identity 的增益 |
| 单步/累计奇异谱 | 指定坐标度量下的 transport 放大与收缩 | 与分支写入、未来读出联合后的实际信号损失 |
| stream pairwise cosine | 当前流基下的相似性 | 对一般换基不变的有效状态阶 |
| 状态矩阵代数 rank | 可逆左换基下不变的状态维度 | 这些维度是否可达、可观测、对任务必要 |
| Gram 特征值/effective rank | 正交变换下不变的谱形状 | 一般非正交变换下的不变性或任务价值 |
| \(K\)、\(\mathcal H_c\) | 固定分支坐标下的路由与跨切分阶，流 gauge 不变 | 闭环任务性能或重新训练的最小网络规模 |

保留几何指标作为数值和优化诊断。RDM曾提出“预测需求→保护→释放”的机制链，但当前已撤销该主线。冻结、恢复或压缩仍可作为具体候选的归因工具；不能由这些诊断自行推出应该增加记忆管理器。

### 7.2 补空间乘积的实现前提

[collect_transport_report](../../lm/transport_analysis.py) 的历史 schema 1 累计的是

\[
(U^\top H_{t-1}U)\cdots(U^\top H_0U).
\]

它通常不等于 \(U^\top(H_{t-1}\cdots H_0)U\)。当所有 \(H_k\) 共同保留补空间时相等；有交换时，前者在每一步之间投影并删除途经均值的路径。充分条件只需 \(e_0^\top H_kU=0\)，并不要求双向固定均值；因此旧 mHC 的行和误差本身不能证明该复合谱有明显偏差，其列和接近 1 表明补空间近似不变。

反例：在 \([e_0,u]\) 基中取二维旋转 \(R(\theta)\)、\(R(-\theta)\)。完整乘积为 \(I\)，其补空间增益为 1；逐步压缩的乘积却是 \(\cos^2\theta\)。因此一般正交/交换臂应记录完整 transport 乘积，再分析其四块；旧累计字段只能在共同不变子空间条件下解读。对于逐步压缩矩阵 \(B_k\)，未加数值 floor 时最小奇异值之积是其复合最小奇异值的下界，最大奇异值之积是其复合最大奇异值的上界；这些界不自动适用于允许交换的完整路径，平均奇异值之积也没有相同的一般等式或界。

本轮已修复为 schema 2：`composite_sv_*` 取完整乘积后的压缩谱，`projected_step_product_sv_*` 明确保留历史算法。90°交换往返、保均值对照与非交换顺序的[回归测试](../../tests/test_transport_composition.py)通过。旧 JSON 保持原值，不能因代码修复而将其字段追溯解释为新算子。当前诊断使用 float32；极小复合增益需另测更高精度，不能将舍入后的零当作精确秩亏。

### 7.3 静态 head-output mixing 可直接吸收进输出投影

[lm/headmix.py](../../lm/headmix.py) 在每头输出后、[lm/models.py](../../lm/models.py) 的自由稠密 `o_proj` 前应用静态 \(H\)。若每头维度为 \(d_h\)，则

\[
W_o(H\otimes I_{d_h})h=W_o'h,
\qquad W_o'=W_o(H\otimes I_{d_h}).
\tag{19}
\]

这对奇异 \(H\) 也成立，不需要求逆。给定自由输出投影与 identity 选项，这种静态 head mixing 没有增加该模块的函数类；它可能改变优化、正则化和参数化。在投影受结构限制、参数共享或 mixing 输入依赖时，需重新检查闭合性。该位置不能作为“新的 attention head 交互能力”主张的证据。

### 7.4 NS 投影的正确保证

对于精确正交的 \(U\)，任意 raw 矩阵 \(A\) 的最近固定向量正交投影为

\[
\Pi(A)=P_0+U\operatorname{polar}(U^\top AU)U^\top.
\tag{20}
\]

证明：在 \([e_0,U]\) 基下，目标矩阵只有右下角 \(R\in O(n-1)\) 可变；Frobenius 目标分离为常数加 \(\|U^\top AU-R\|_F^2\)，正交 Procrustes 的解为 SVD 极因子。右下块奇异时解可不唯一；实现没有强制 \(\det R=+1\)，所以目标是 \(O(n-1)\) 而非全局 \(SO(n-1)\)。因此在 \(n=4\) 时，oHC 讨论的保均值 \(SO(3)\) 与这里的恒等连通分支一致，不能将二者的完整可行集无条件视为相同。

NS 对奇异值的更新为 \(s\mapsto s(3-s^2)/2\)。Frobenius 缩放将非零初始奇异值置于 \((0,1]\)，在精确算术下迭代趋向 1；零奇异值始终为零，不能单靠增加 NS 步数修复秩亏。有限步仅近似等距，SVD fallback 也涉及非唯一点与数值反传问题。

当前底层投影内部使用 float64，但调用方预计算的 \(U\) 常为 float32，再转换为 float64 不会恢复已经损失的精度；`lm/mixing.py` 的“fp32”文字说明与底层实现不一致。报告应以实测固定向量/正交误差及实际执行路径为准。

若每步补空间误差满足 \(\|\hat B_k^\top\hat B_k-I\|_2\le\epsilon_k<1\)，则复合 transport 的奇异值受

\[
\prod_k\sqrt{1-\epsilon_k}
\le\sigma_{\min}(\hat B_{N-1}\cdots\hat B_0)
\le\sigma_{\max}(\hat B_{N-1}\cdots\hat B_0)
\le\prod_k\sqrt{1+\epsilon_k}
\tag{21}
\]

约束。这解释了为什么单步很小的误差仍需随深度核查；它仍不是整个非线性网络的稳定性保证。

## 8. 现在可以写入论文、需要验证、需要删除的陈述

| 级别 | 陈述 | 后续证据责任 |
|---|---|---|
| 已证明 | 静态可逆 transport 可以代数消去，但架构等价需要参数类与边界闭合 | 小型精确等价检查，包含最终出口、负例和奇异例外 |
| 已证明 | 当前参数类在精确 IsoHC 约束下与 identity-HC 函数类相同，正交读写重表示不增加欧氏范数 | 数值误差、训练过程与正则化另行报告 |
| 已证明 | kernel 在流 gauge 下不变；跨切分 rank 至多 \(n\) | 用真实 checkpoint 的读写/顺序构造，不以几何谱代替 |
| 已证明 | 保均值 identity-HC 也能实现大于 1 的有效路由阶 | (15) 是构造性反例；任务必要性尚未证明 |
| 已撤销候选的假说 | 预测后续读取需求改善保留/释放 | 没有训练或工程证据；不是当前主方法 |
| 辅助诊断假说 | 实际多流模型的部分状态自由度可压缩 | 持出数据与受控干预；不作为新方法收益的替代 |
| 可检验假说 | 某些动态/受限读写模型需要更多条件化路由能力 | 保持同源架构，比较动态 read/write、transport 与参数预算 |
| 可检验假说 | 非等距均值增益解释部分小模型优势 | 单流可学习 residual gain、出口读出、初始化和充分训练对照 |
| 应删除 | 三种 transport 机制穷尽额外状态价值，或已穷尽 transport 的非等价来源 | (15) 反驳前者；后者遗漏 §6 的参数与边界限制 |
| 应删除 | 动态 IsoHC 必须等于 identity，或动态必然破 gauge | 前者没有闭合证明，后者有可消去反例 |
| 应删除 | 保住 complement 能量即保住有用记忆 | 未经过读写与任务检验 |
| 应删除 | \(\lambda=0.01\) 因而所有干预的 loss 变化必为 \(10^{-4}\) | 误把初始化核系数缩放当全网误差界 |
| 应删除 | oHC 的任务增益已被证明只来自 mean/difference 交换 | [原论文](https://arxiv.org/html/2609.02672v1) 的几何分析与干预不足以完成这一排他因果归因 |

这些结果是审计基础，不构成已完成的架构贡献。RDM的需求预测与保护/释放组合已撤销主线资格；其后续数学记录只说明特定算子的性质，不能自动触发该方案开发。

## 9. 保留—复用与受保护槽位：已撤销候选的理论记录

### 9.1 同分支源下，等距保留与有限状态复用存在冲突

本节只比较固定分支源的辅助路由，不证明任意重训网络的能力下界。令辅助维度为 \(r=n-1\)，初始状态 \(Y_0=0\)，写入与随后读取的时序为

\[
Y_{i+1}=A_iY_i+v_i\delta_i,\qquad s_i=u_i^\top Y_{i+1},\qquad
D_{ij}=u_i^\top(A_i\cdots A_{j+1})v_j\quad(j\le i),
\tag{22}
\]

其中 \(i=j\) 时乘积为 \(I\)。在保均值 HC 中，取 \(u_i=U^\top a_{i+1}/\sqrt n\)、\(v_i=U^\top b_i/\sqrt n\)，并相应缩放辅助状态，则 \(D_{ij}=K_{i+1,j}-1\)：标准均值累加通道仍保留。

**命题 4：精确复用的等距状态下界。** 若所有 \(A_i\) 正交，连续 \(T\) 次读出都满足 \(D_{ii}=1\)、\(D_{ij}=0\;(j<i)\)，则 \(r\ge T\)。证明：按 §3 作正交 gauge 后，\(D_{ij}=\widetilde u_i^\top\widetilde v_j\)。由全部读写构成的 \(T\times T\) 完整矩阵，其对角为 1、严格下三角为 0，故秩为 \(T\)；其读写分解的中间维度只有 \(r\)。未观测的严格上三角元素任意，不改变该结论。

**有界近似版。** 若 \(\|u_i\|\le C\)、\(\|v_i\|\le B\)、\(D_{ii}\ge1\)，且 \(\sum_{j<i}D_{ij}^2\le\epsilon^2\)，则对任意 \(\tau>0\)，

\[
T\log\!\left(1+\frac1{\tau C^2+\epsilon^2}\right)
\le r\log\!\left(1+\frac{TB^2}{r\tau}\right).
\tag{23}
\]

证明：在正交 gauge 中设 \(V_i=\tau I+\sum_{j<i}\widetilde v_j\widetilde v_j^\top\)。Cauchy–Schwarz 给出 \(\widetilde v_i^\top V_i^{-1}\widetilde v_i\ge1/(\tau C^2+\epsilon^2)\)；矩阵行列式引理给出行列式逐步增长下界，\(\operatorname{tr}V_T\le r\tau+TB^2\) 与特征值的算术—几何平均不等式给出上界，合并即得 (23)。固定正的范数预算和固定精度时，此式要求 \(r=\Omega(T)\)。若创新源独立、零均值且单位方差，泄漏平方和就是旧源造成的读出方差；自然文本的分支输出未必满足该假设，不能直接套用为任务损失。

相对地，单个辅助状态取 \(A_i=\rho\)、\(u_i=v_i=1\)，则 \(D_{ij}=\rho^{i-j}\)，任意深度的旧源泄漏至多 \(\rho^2/(1-\rho^2)\)。取 \(\rho=0\) 可精确反复复用。在两流 HC 中，\(H=P_0+\rho P_\perp\)、\(a_{i+1}=b_i=(2,0)^\top\)、\(0\le\rho<1\) 就实现这一构造，且读写非负、满足 fixed-sum。收缩同时会毁掉仍需长期读取的信息，因此该构造只说明**应依据后续需求决定保留还是释放**，不证明“收缩总是更好”。

上述线性代数与行列式势是经典工具，HC 特化构造也尚未完成首次性核验。静态 IsoHC 与 identity-HC 在相同欧氏范数预算下仍然等价；(23) 区分的是等距保留与允许遗忘的同源路由。训练后的分支可以主动抵消旧内容，实际收益需由完整训练证明。

### 9.2 主实现合同：保护槽位，受限替换

候选架构维护 active workspace 和按槽位存储的深度记忆 \(M\in\mathbb R^{n\times d}\)。此处的“正交槽位”仅指槽位坐标轴，不要求各行的语义向量正交。控制器依据当前可用信息生成保护掩码 \(p\in\{0,1\}^n\)、写入许可和释放决定。分支产生经过明确范数约束的 proposal \(v\)，\(\|v\|_2\le R\)。每次只更新一个未受保护槽位 \(j\)，

\[
p_j=0,\qquad M'_j=(1-\eta)M_j+\eta v,\quad 0\le\eta\le1,
\qquad M'_i=M_i\ (i\ne j).
\tag{24}
\]

若所有槽位都受保护，历史 RDM 版本规定拒绝本步持久写入，active workspace 仍接收 proposal；仅到期保护可解除，不提前撤销已承诺的 lease。允许主动提前释放的未来变体必须另行修改架构合同，不能暗中覆盖。式 (24) 的 proposal 约束、初始化和控制范围必须由实现满足。

**命题 5：受保护写入的局部保证。** (a) 对任意只支持在受保护槽位上的读向量 \(a\)，\(a^\top M'=a^\top M\)；(b) 若初始每行范数不超过 \(R\)，所有后续行范数仍不超过 \(R\)；(c) 固定控制、槽位选择与 proposal 时，更新对 \(\operatorname{vec}M\) 的 Jacobian 的谱范数至多 1。证明分别来自未改动坐标、范数凸性，以及对角线为 1 或 \(1-\eta\) 的槽位算子。此保证只涵盖记忆更新，不涵盖 active workspace 的演化。

多 proposal 的合法扩展为 \(M'_i=(1-\sum_h\eta_{ih})M_i+\sum_h\eta_{ih}v_h\)，其中 \(\eta_{ih}\ge0\)、每行 \(\sum_h\eta_{ih}\le1\)，受保护行的门全部为零；相同证明成立。若 proposal、门或选择依赖状态，真实 Jacobian 还包含其导数；硬选择的切换边界甚至可能不可微。**(24) 不保证整个网络非扩张、梯度稳定或模型质量更好。**

### 9.3 后续需求预测如何约束释放损害

RDM历史提案的机制是学习未来读取需求的 **lease（有限保护承诺）、admission（写入许可）和 release（解除保护）**。预测器推理时只能使用当前及此前状态；训练可以使用停止梯度的后续读取记录作为监督，但必须区别训练监督与推理可见信息。DNC 已有内容寻址、分配、释放和复用，[RMT](https://arxiv.org/abs/2506.22696) 已有扩展残差记忆，[DDL](https://arxiv.org/abs/2601.00417) 已有定向擦写；这些概念和 (24) 的凸更新本身不是创新。[DNC 原论文](https://www.nature.com/articles/nature20101)

给定一次写入决策，冻结后续控制与 proposal，令 \(a_{tj}\) 是当前槽位 \(j\) 到后续读出 \(t\) 的**有效**系数。如果期间该槽位再被 (24) 更新，系数须包含各次 \(1-\eta\) 的乘积；没有后续更新时，它才等于直接读权重。替换 \(j\) 造成 \(\Delta M_j\)，则加权读出平方变化精确为

\[
D_j=\sum_{t>k}w_t a_{tj}^{\,2}\|\Delta M_j\|_2^2,\qquad w_t\ge0.
\tag{25}
\]

这不是 loss 变化公式。对固定写入目的和同一合法候选集合 \(\mathcal A\)，若预测逐候选满足 \(|\widehat D_j-D_j|\le\varepsilon\)，令 \(\widehat j\in\arg\min_{j\in\mathcal A}\widehat D_j\)，则

\[
D_{\widehat j}-\min_{j\in\mathcal A}D_j\le2\varepsilon.
\tag{26}
\]

证明由预测值与真实值之间两次至多 \(\varepsilon\) 的偏差及择优条件直接得到。这是标准择优误差界，不是新的学习理论；它没有证明预测器能达到该误差。候选必须满足共同的写入需求，否则“永不写入”以零损害平凡获胜。真实评估必须重新计算后续读取、proposal 和控制，并报告任务质量；冻结路径的 (25)–(26) 只承担可测机制合同。

### 9.4 为什么主实现不采用一般斜投影写入

一般 dual/nullspace/RLS 写入 \(M'=M+v(q^\top M_{\mathrm{target}}-q^\top M)\) 即使满足 \(q^\top v=1\)，其旧状态算子 \(I-vq^\top\) 也可非正规放大。取 \(q=(1,0)^\top\)、\(v=(1,L)^\top\)，则算子为 \(\left(\begin{smallmatrix}0&0\\-L&1\end{smallmatrix}\right)\)，特征值只有 0 和 1，但谱范数为 \(\sqrt{1+L^2}\)。因此“实现目标读出且保住某些 views”不能替代算子范数合同。此前 RDM 提案因此选择 (24) 的受保护槽位更新；该段是历史设计理由，不指定当前实现。

RDM已撤销主线资格，完整模型与预测器尚未实现、训练；[独立原语核验](../../results/depth_memory_contracts_20260925/README.md)只检查本节声明的局部合同。这些局部命题不支持恢复RDM优先级；主线仍应先满足最小原语、标准训练与真实工程合同。

## 10. R4：严格 adjoint 两流残差

### 10.1 确定的更新规则

维护 \(X_\ell=(x_\ell,m_\ell)^\top\in\mathbb R^{2\times d}\)，分别为 primary 与辅助流。每个 attention/MLP 分支各有一个连续 router；逐 token 计算

\[
t_\ell=\tanh\!\left(b_\ell+
\frac{w_\ell^\top\operatorname{RMSNorm}(x_\ell)}{\sqrt d}\right),\qquad
c_\ell=\frac{(1,t_\ell)^\top}{\sqrt{1+t_\ell^2}},\qquad
G_\ell=F_\ell\circ\operatorname{Norm},
\tag{27}
\]
\[
z_\ell=c_\ell^\top X_\ell,\qquad
X_{\ell+1}=X_\ell+c_\ell G_\ell(z_\ell).
\tag{28}
\]

也就是 \(x'=x+G(z)/\sqrt{1+t^2}\)、\(m'=m+tG(z)/\sqrt{1+t^2}\)，同一单位向量既读又写。主规则没有独立 write 向量、transport 矩阵、投影迭代、目标替换或额外监督；当前输出 head 使用同形式的可学习单位地址读出，初始化为 primary。router、读出和写回均为规则的 \(O(d)\) 运算；两流状态的激活、带宽和实际耗时成本仍须测量。有限的 \(t\) 范围是实际模型约束，不得当成任意单位方向类。

### 10.2 冻结 routing 后，正交补保留与局部谱是精确的

固定单位读向量 \(c\)，令 \(z=c^\top X\)、\(X_\perp=(I-cc^\top)X\)。式 (28) 精确等于

\[
z'=z+G(z),\qquad X'_\perp=X_\perp,
\qquad
\|X'\|_F^2-\|X\|_F^2
=\|z+G(z)\|_2^2-\|z\|_2^2.
\tag{29}
\]

即分支修改它刚刚读取的那个 view，其他正交 view 不变；能量变化正好是该普通 residual 分支的能量变化。该式不保证能量有界，普通 residual 本身仍可放大。不同层的 \(c\) 不同，也不产生一个全深度共同受保护的补空间。

记 \(J=DG(z)\)，其中包括实际 Norm 的导数。冻结 routing 后的局部 Jacobian 在首轴为 \(c\) 的正交基下为

\[
\mathcal J_{\rm adj}
\sim_{\rm orth}\operatorname{diag}(I_d+J,I_{(n-1)d}).
\tag{30}
\]

因此其奇异值恰为 \(I_d+J\) 的奇异值加上 \((n-1)d\) 个 1，而非只得到 carry 等距。这里写成一般 \(n\ge2\) 以陈述算子性质；R4 实现取 \(n=2\)。对 attention，将全序列特征展平：冻结的逐 token 单位读向量组成 coisometry \(C\)，满足 \(CC^\top=I\)，adjoint 写回为 \(C^\top\)，同样得到 (30)。这覆盖 attention 的跨 token 导数，不要求 \(J\) 逐 token 分块。

**命题 6：同读取、同分支及单位自作用下，adjoint 写回局部条件最优。** 比较所有满足 \(c^\top b=1\) 的冻结 write \(b\)，保持同一 \(c,z,G,J\)。写成 \(b=c+u\)、\(u\perp c\)，则其 Jacobian 在相同坐标分解下为

\[
\mathcal J_b\sim_{\rm orth}
\begin{pmatrix}A&0\\B&I\end{pmatrix},\qquad
A=I_d+J,\quad B=u_\perp\otimes J.
\tag{31}
\]

式 (31) 的 Kronecker 形式用于共享 \(c\)；逐 token \(c\) 的 coisometry 情形仍有相同块形式，\(B\) 替换为补空间写回算子与 \(J\) 的复合。限制输入到 active 子空间和补空间，分别给出 \(\sigma_{\max}(\mathcal J_b)\ge\|A\|_2\) 与 \(\sigma_{\max}(\mathcal J_b)\ge1\)。adjoint 的 \(B=0\) 达到下界 \(\max(\|A\|_2,1)\)。若 \(A\) 可逆，逆矩阵为 \(\left(\begin{smallmatrix}A^{-1}&0\\-BA^{-1}&I\end{smallmatrix}\right)\)，同理可得 \(\sigma_{\min}(\mathcal J_b)\le\min(\sigma_{\min}(A),1)\)，adjoint 达到上界；若 \(A\) 奇异，所有比较算子均奇异。因此 adjoint 同时最小化最大奇异值、最大化最小奇异值，并在非奇异情形最小化局部 2-范数条件数。

这是指定比较类中的逐点结论，不是跨架构、跨参数点、整网或优化过程的普遍最优性。一般 HC 可以选择不同读取、分支输入、有效增益和任务解；命题不证明限制成 adjoint 后质量必然更好。其直接含义是：保持这些对象不变时，读写错位只能引入额外的 shear，不能改善这里的极端奇异值合同。证明是经典块矩阵线性代数的应用，首次性并未建立。

实际 R4 的 \(c=c(X)\) 可学习且依赖状态。按 (28) 将 \(G(z)\) 写成行向量，真实微分除 (30) 外，还含
\(dc\,G(z)+c\,DG(z)[(dc)^\top X]\)，其中 \(dc=Dc(X)[dX]\)。冻结 routing 的最优性不控制这些路由导数，不保证全网梯度稳定；训练后须单独测量实际 Jacobian 或相应扰动响应。

### 10.3 Signed carrier：前向严格 baseline，入口一阶，核仍二阶

取固定 \(D=\operatorname{diag}(s_1,\ldots,s_d)\)，\(s_i\in\{-1,+1\}\) 且含两种符号。初始化

\[
X_0=(x_0,Dx_0)^\top,\qquad b_\ell=0,\quad w_\ell=0\quad\text{对所有分支}.
\tag{32}
\]

此时所有 \(t_\ell=0\)、\(c_\ell=e_1\)，由归纳得 \(x_{\ell+1}=x_\ell+G_\ell(x_\ell)\)、\(m_\ell=Dx_0\)。使用相同 branch、归一化、出口及耦合的 dropout，初始函数严格等于标准 residual baseline。\(D\) 不增加输入信息、没有可训练参数，且 \(\|Dx_0\|=\|x_0\|\)；它提供一个初始不可见、通常非共线的 carrier：当 \(x_0\) 在两种符号组均有非零分量时，\(Dx_0\) 与 \(x_0\) 非共线。这不意味着已经学到了辅助记忆。

对某层的初始 router 标量 \(t_\ell\) 做局部变化，有 \(\partial z_\ell/\partial t_\ell=m_\ell\)、\(\partial x_{\ell+1}/\partial t_\ell=DG_\ell(x_\ell)m_\ell\)。其写入辅助流的一阶变化此时不会被后续零 router 的 primary 出口读到。设 baseline 下游伴随梯度为 \(g_{\ell+1}\)，则

\[
\left.\frac{\partial\mathcal L}{\partial t_\ell}\right|_{t=0}
=\langle DG_\ell(x_\ell)^\top g_{\ell+1},Dx_0\rangle.
\tag{33}
\]

对 attention，(33) 按全部 token 的伴随计算后逐 token 取内积；再乘 router 特征即可得到 \(w_\ell\) 的梯度。该量一般可以非零，但分支导数为零、数据对称或特定正交关系仍会使其为零；不能承诺每层每个 batch 都有非零梯度。全零辅助初始化则在全零 router 下精确退化为路由梯度死区。signed carrier 避免把种子仅做成当前读出的一次整体缩放，后者可能被归一化的尺度不变性消去。

更关键的边界是：冻结一次 routing 轨迹后的同源深度核仍为

\[
K_{tj}=c_t^\top c_j
=\frac{1+t_t t_j}{\sqrt{(1+t_t^2)(1+t_j^2)}}
=1-\tfrac12(t_t-t_j)^2+O(\|(t_t,t_j)\|^4).
\tag{34}
\]

在初始化点所有核的角导数仍为零；(33) 的一阶信号来自入口 carrier 项 \(c_t^\top X_0\)，不是新深度核的一阶变化。对任意相同分支源要求单位 tied 读写精确复现标准 residual，会要求每个相邻 \(c_{j+1}^\top c_j=1\)，从而迫使这些单位向量相同；这个核的二阶起点不是靠新名称就能消除的。carrier 能先推动路由离开共线点，但是否形成有用记忆必须由训练证明。

### 10.4 算法贡献的实际边界

R4 是 [一般 HC](https://arxiv.org/html/2409.19606) 读写模型的受限子类；[RMT](https://arxiv.org/html/2506.22696) 已有向量检索与外积写回。[Stream Collapse](https://arxiv.org/html/2606.03483v1) 已用逐特征流初始化破对称，因此 signed carrier 也不能独立声称为首次。将 (28) 改成 \(X'=X+\beta c(v-c^\top X)\) 会进入已有 [DDL](https://arxiv.org/abs/2601.00417) 式目标擦写思路，不属于当前规则。

[CliffSearch](https://cliffsearch.ai/assets/cliffsearch_preprint.pdf) 的公开 HC 搜索资产还包含 GrassmannianSubspaceRouting 的投影—提升机制；其导出代码与严格 tied 更新、动态路由、初始化及理论合同的逐式比较尚待完成。[原文 Table 6 与导出材料](https://cliffsearch.ai/assets/cliffsearch_preprint.pdf)将 raw-best 节点 G3/H2 判为跨样本泄漏无效，同名另一节点 H1 通过了该项审计，不能按 alias 混用有效性或把 raw-best 数字当成有效质量基线。机制上的邻近性仍需核查，当前不能声称 R4 的单位读写原语或局部谱结论首次提出。所选算法值得实现和公平训练，但竞争力、动态路由的净收益、carrier 是否主要充当 embedding 长 skip，以及相对强 iHC/mHC/AttnRes 的实际收益，均仍待验证。
