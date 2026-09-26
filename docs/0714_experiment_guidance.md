<!-- doc-status: historical -->
> **Historical user source from 2026-07-14:** retained as context. Its hypotheses,
> implementation directions, and tool instructions are not current guidance.
> Follow the [current research owner](research/README.md) and
> [current experiment plan](research/roadmap.md).

<!-- doc-history-body-begins -->
# IsoHC 实验指导（工作假设）

> **文档性质：** 这是实验设计的参考与待验证假设，不是必须奉行的真理，也不是对当前代码、相关论文或未来结果的最终裁决。
>
> **使用原则：** 每项判断都应分别用当前仓库代码、原始引用、可复现实验和统计结果重新核验；证据不支持时，应修改或放弃本指导中的建议。
>
> **来源与保真：** 正文来自 2026-07-14 用户提供的 pasted text。除纯等号公式行前添加 Markdown 转义以避免被 Git 误判为冲突标记外，其余正文按原文保存。保存为 Markdown 不代表其中结论已经得到仓库或外部来源验证；粘贴过程中形成的公式排版也保留原状，投稿前需单独校正。

---

## 原始指导正文

你说得对。上一版没有以仓库的一手内容为依据，这是我处理不当。现在我已经按链接核查了仓库的 README、论文 `paper/isohc_arxiv/main.tex`、`isohc/projection.py`、`lm/mixing.py`、`lm/models.py`、主实验 runner、机制分析脚本以及 48-layer FineWeb-Edu、24-layer LM、GNN 等实验报告。

重新看完以后，我的判断比上一版更具体：

> 你们目前的工作已经是一篇逻辑自洽、结论克制的 **mechanism paper**：
> **mHC/Birkhoff 保住 residual mean，却可能系统性耗散 mean-zero complement；IsoHC 用 fixed-vector orthogonal transport 修复这一点。**
>
> 但若目标是 CCF-A 主会长文，当前最大的缺口不再是“正交性证明不够”，而是：
>
> 1. 当前 LM 中的 `mHC` 是静态 Birkhoff 代理，并非原始 mHC 的动态实现；
> 2. 目前的巨大 complement contraction 很大程度已由 `diag_bias=4` 和 96 次 transport 预置；
> 3. IsoHC 保住了 complement，但现有 read/write 接口没有证明模型真正使用了它；
> 4. 静态可逆 mixer 存在很强的 gauge equivalence，单纯 learned rotation 不一定扩展函数类；
> 5. “orthogonal HC”本身已有直接并行工作，下一版 novelty 必须推进到 **可控、可观、双边稳定的 residual state-space transport**。

---

# 一、先准确描述仓库里现在真正实现的模型

仓库主 LM 不是一般的动态 HC，而是一个两分支静态 HC Transformer。每层 Attention 和 MLP 各执行一次 residual-stream transport，因此 48 层对应 **96 个 transport operators**：

[
X_{l+\frac12}
\=============

H_l^{\mathrm{attn}}X_l
+
b_l^{\mathrm{attn}}
\otimes
\mathrm{Attn}!\left(
\mathrm{LN}\left(
\frac{(a_l^{\mathrm{attn}})^\top X_l}{n}
\right)\right),
]

[
X_{l+1}
\=======

H_l^{\mathrm{mlp}}X_{l+\frac12}
+
b_l^{\mathrm{mlp}}
\otimes
\mathrm{MLP}!\left(
\mathrm{LN}\left(
\frac{(a_l^{\mathrm{mlp}})^\top X_{l+\frac12}}{n}
\right)\right).
]

这里 (H_l,a_l,b_l) 都是每层静态可学习参数，不随 token 或输入变化。代码还使用零均值 `stream_embed` 打破初始流对称性，并将

[
a_l=\mathbf1+\lambda_a(w_l-\bar w_l\mathbf1),
\qquad
b_l=\mathbf1+\lambda_b(v_l-\bar v_l\mathbf1)
]

中的 (\lambda_a,\lambda_b) 初始化为 (0.01)，而且当前是所有层、两个分支共享的全局标量。([GitHub][1])

这和原始 HC/mHC 有重要区别：原始方法的 read、write 和 residual mappings 都包含 input-dependent dynamic 部分与 static 部分；mHC 再对动态 residual mapping 做 Sinkhorn 投影，并对 read/write 施加正约束。([arXiv][2])

因此当前代码里的方法更准确的名字应当是：

* `static identity-HC`
* `static unconstrained-HC`
* `static Sinkhorn/Birkhoff-HC`
* `static fixed-vector IsoHC`

当前实现作为受控几何实验是合理的，但不应直接声称已经和官方 dynamic mHC 做了完整对比。

---

# 二、从 residual connection 的本质重新推导当前 IsoHC

## 2.1 residual 的本质是“持久状态”，不是简单的加号

令单个 Attention/MLP 子步骤统一写成

[
z_k=\frac1n a_k^\top X_k,
\qquad
y_k=F_k(N_k(z_k)),
]

[
X_{k+1}=H_kX_k+b_ky_k^\top,
\tag{1}
]

其中 (X_k\in\mathbb R^{n\times d})。

定义

[
v=\frac1{\sqrt n}\mathbf1,\qquad
P=vv^\top=\frac1n\mathbf1\mathbf1^\top,
]

并取

[
U\in\mathbb R^{n\times(n-1)},
\qquad
U^\top U=I,\quad U^\top\mathbf1=0.
]

任意 stream state 唯一分解为

[
X_k
===

\mathbf1\mu_k^\top+UC_k,
\tag{2}
]

其中

[
\mu_k=\frac1n\mathbf1^\top X_k\in\mathbb R^d
]

是普通 residual stream，而

[
C_k=U^\top X_k\in\mathbb R^{(n-1)\times d}
]

是多出来的 stream-complement memory。

如果要求

[
H_k\mathbf1=\mathbf1,
\qquad
\mathbf1^\top H_k=\mathbf1^\top,
\qquad
\mathbf1^\top a_k=\mathbf1^\top b_k=n,
\tag{3}
]

定义

[
B_k=U^\top H_kU,\qquad
\alpha_k=\frac1nU^\top a_k,\qquad
\beta_k=U^\top b_k,
]

则式 (1) 精确化为

[
z_k=\mu_k+\alpha_k^\top C_k,
\tag{4}
]

[
\boxed{\mu_{k+1}=\mu_k+y_k,}
\tag{5}
]

[
\boxed{C_{k+1}=B_kC_k+\beta_ky_k^\top.}
\tag{6}
]

这是你们工作最应该置于论文核心位置的 canonical form。

它说明 HC 并不是“多份 residual 相互混合”这么简单，而是一个低维控制系统：

* (\mu_k)：原始 residual identity state；
* (C_k)：额外的隐式 memory state；
* (B_k)：memory transport；
* (\alpha_k)：memory read/observability interface；
* (\beta_k)：memory write/controllability interface。

因此，一个真正完整的现代 HC 需要同时满足三个条件：

[
\boxed{
\text{mean identity}
+
\text{stable complement transport}
+
\text{complement accessibility}.
}
]

当前 IsoHC 解决了第二项，但第三项尚未被证明。

---

## 2.2 mHC 为什么不是“可能”扩散，而是在 Birkhoff 内部必然严格扩散

论文当前已经证明：

[
H\in\mathcal B_n
\quad\Longrightarrow\quad
|U^\top HU|_2\le1,
]

以及

[
H_\rho
======

\rho I+(1-\rho)P
]

会在 complement 上产生 (\rho^L) 收缩。这个反例是正确的。([GitHub][3])

还可以把它加强成一个更有杀伤力的定理。

### 定理：Birkhoff 内点的严格 complement contraction

设

[
H\in\mathcal B_n,
\qquad
H_{ij}\ge\eta>0.
]

因为每行和每列和为 1，可以写成

[
H
=

n\eta P+(1-n\eta)\widetilde H,
]

其中

[
\widetilde H\in\mathcal B_n.
]

对任意 (x\perp\mathbf1)，有 (Px=0)，所以

[
Hx=(1-n\eta)\widetilde Hx.
]

由双随机矩阵的二范数不超过 1：

[
\boxed{
|Hx|_2
\le
(1-n\eta)|x|_2.
}
\tag{7}
]

若每层最小元素为 (\eta_k)，则

[
\left|
B_{K-1}\cdots B_0
\right|*2
\le
\prod*{k=0}^{K-1}(1-n\eta_k)
\le
\exp\left(-n\sum_k\eta_k\right).
\tag{8}
]

这意味着：

> 只要 Sinkhorn 输出停留在 Birkhoff polytope 的严格内部，transport-only complement 就必然严格收缩。
> 要避免长期耗散，它必须趋向 Birkhoff 边界，接近 permutation。

因此 mHC 的几何本质是一个 **Markov/diffusive semigroup**：

* 均值是 stationary mode；
* complement 是 mixing modes；
* 非负 mixing 的功能正是把不同 stream 做 consensus。

IsoHC 则是 conservative/unitary transport：

[
B_k^\top B_k=I.
]

这两者不是“哪个一定更好”，而是：

* mHC 擅长融合、去噪、遗忘；
* IsoHC 擅长持久记忆、可逆传输；
* 大模型最可能需要的是二者之间的可控区域。

---

## 2.3 当前 48 层结果其实非常自洽

仓库报告中，48 层模型每层有 Attention 和 MLP 两次 transport，共 96 次。训练后 mHC 单层 complement singular values 约为

[
\bar\sigma_\perp=0.9222,
]

而 IsoHC 约为

[
\bar\sigma_\perp=0.99996.
]

直接计算：

[
0.9222^{96}
\approx
4.20\times10^{-4},
]

而实际 composite gain 是

[
4.39\times10^{-4}.
]

同样：

[
0.99996^{96}
\approx0.99617,
]

实际 IsoHC composite gain 是

[
0.996349.
]

这几乎精确地解释了实验结果，说明你们测到的 composite contraction 不是指标噪声，而是真正的深度复合谱效应。([GitHub][4])

但这里也暴露出最重要的实验混淆因素。

当前 `MHCMixing` 使用

[
\texttt{diag_bias}=4
]

初始化 Sinkhorn logits。忽略小噪声时，初始矩阵为“对角为 (e^4)、非对角为 1”归一化后的对称双随机矩阵，其 complement eigenvalue 为

[
\rho_0
======

\frac{e^4-1}{e^4+n-1}.
]

当 (n=4) 时：

[
\rho_0\approx0.93055,
]

于是仅由初始化重复 96 次就有

[
\rho_0^{96}\approx9.98\times10^{-4}.
]

也就是说，即使完全不训练，当前参数化就已经预置了千分之一量级的 composite contraction。训练只是把它进一步推到 (4.39\times10^{-4})。`diag_bias`、temperature 和 noise 当前也没有暴露为主实验参数。([GitHub][5])

所以当前数据证明了：

> “这种静态 Sinkhorn/Birkhoff 参数化在 96 次复合后发生强 contraction。”

但还没有完全证明：

> “一个忠实、动态、充分训练的 mHC 在合理初始化和大模型训练中不可避免地发生同等程度的 contraction。”

这是必须补的因果对照。

---

# 三、当前 IsoHC 只解决 transport isometry，不解决完整 block stability

将 token 和 hidden 维度统一展平为 (D)，记

[
J_k=D(F_k\circ N_k)(z_k).
]

由式 (1)，完整子步骤 Jacobian 是

[
\boxed{
D\Phi_k
\=======

H_k\otimes I_D
+
\frac1n(b_ka_k^\top)\otimes J_k.
}
\tag{9}
]

在 mean/complement 坐标下则更清楚：

[
D\Phi_k
\=======

\begin{bmatrix}
I+J_k
&
J_k(\alpha_k^\top\otimes I_D)
[2mm]
(\beta_k\otimes I_D)J_k
&
B_k\otimes I_D+
(\beta_k\alpha_k^\top)\otimes J_k
\end{bmatrix}.
\tag{10}
]

式 (10) 表明：

1. IsoHC 只使左下角 transport baseline 中的 (B_k) 正交；
2. mean path 上仍然是标准 residual Jacobian (I+J_k)；
3. complement 与 branch 之间的耦合由 (\alpha_k,\beta_k) 决定；
4. 即使 (B_k) 完全正交，完整 block 也未必是 dynamical isometry。

若

[
\epsilon_k
\==========

\frac{|a_k|_2|b_k|_2}{n}|J_k|_2,
]

则对静态 IsoHC：

[
\sigma_{\min}(D\Phi_k)\ge1-\epsilon_k,
]

[
\sigma_{\max}(D\Phi_k)\le1+\epsilon_k.
\tag{11}
]

只有当 residual branch perturbation 本身受控时，完整 block 才接近等距。你们当前论文已经诚实地把这一点列为 limitation：IsoHC 是 transport component isometry，而非整个 Transformer block 的 dynamical isometry。([GitHub][3])

这里要修正我上一版的一点：当前代码里的 (H_k) 是静态参数，所以不存在

[
DH_k(X)[\Delta X]X
]

这一动态-router Jacobian 项。只有下一步引入忠实的 input-dependent IsoHC 时，这一项才必须进入理论。

---

# 四、为什么当前 complement 被保留了，却几乎没有功能作用

这可以从式 (4)–(6)直接解释。

当前初始化中

[
\lambda_a=\lambda_b=0.01.
]

于是

[
|\alpha_k|=O(\lambda_a),
\qquad
|\beta_k|=O(\lambda_b).
]

因此：

* complement 对 branch input 的影响为 (O(\lambda_a))；
* branch 向 complement 的写入为 (O(\lambda_b))；
* “写入 complement，再经后层读出”的闭环影响为

[
O(\lambda_a\lambda_b)
\=====================

O(10^{-4}).
]

更强地，当

[
\alpha_k=0,\qquad
\beta_k=0,
]

且最终 readout 也只取均值时：

[
z_k=\mu_k,
]

[
\mu_{k+1}
\=========

\mu_k+F_k(N_k(\mu_k)),
]

[
C_{k+1}=B_kC_k,
]

但 logits 完全不依赖 (C_k) 或 (B_k)。

即：

> complement 可以被完美保存，但对模型输出完全不可观测。

当前 posthoc 结果正符合这个预测：mHC 和 IsoHC 的 composite gains 相差约三千倍，但 complement removal、IsoHC→identity 和 IsoHC→random-Iso 的验证损失变化都只有 (10^{-4})–(10^{-3})。论文对此也作了正确、克制的解释。([GitHub][6])

所以当前真正需要解决的不是“如何进一步让 (\sigma_\perp=1)”，而是：

[
\boxed{
\text{如何让被保存的 complement 成为可写、可读、可用的状态。}
}
]

---

# 五、一个对当前静态 IsoHC 非常关键的 gauge equivalence

这是我认为你们下一版理论里必须主动加入的结果。

考虑任意静态、可逆的 transport 序列：

[
X_{k+1}
\=======

H_kX_k
+
b_kf_k!\left(\frac1n a_k^\top X_k\right).
\tag{12}
]

定义

[
G_0=I,\qquad
G_{k+1}=H_kG_k,
]

并作坐标变换

[
X_k=G_kZ_k.
]

代入式 (12)：

[
G_{k+1}Z_{k+1}
\==============

G_{k+1}Z_k
+
b_kf_k!\left(
\frac1n a_k^\top G_kZ_k
\right).
]

于是

[
\boxed{
Z_{k+1}
\=======

Z_k+
\widetilde b_k
f_k!\left(
\frac1n\widetilde a_k^\top Z_k
\right),
}
\tag{13}
]

其中

[
\widetilde a_k=G_k^\top a_k,
\qquad
\widetilde b_k=G_{k+1}^{-1}b_k.
]

如果 (H_k) 都保持 (\mathbf1)，那么变换后的 (\widetilde a_k,\widetilde b_k) 仍满足 sum constraint。

### 结论

在一条没有环的前馈深度链上，只要：

* (H_k) 是静态可逆的；
* 每层 read/write vector 足够自由；
* 最终 readout 也可自由变换；

那么任意静态 transport 都可以被 gauge-transform 成 identity transport。

换句话说：

> 当前 static IsoHC 和 identity-HC 在理想参数空间下很可能具有相同的函数类。
> IsoHC 的作用主要是优化坐标系、条件数和归纳偏置，而不是新增函数表达力。

这也解释了为什么 identity-HC 是如此强的对照，以及为什么直接把训练后的 (Q_k) 替换成 identity 只产生很小的损失差。

这个结论并不会削弱 IsoHC，反而会让论文更严谨：

* mHC 的问题变成累计 gauge 的病态条件数；
* IsoHC 选择的是 metric-compatible、condition-number-1 的 gauge；
* 若要证明 learned rotations 有不可消去的功能价值，需要 input-dependent (H_k(X))、跨层参数共享、拓扑环路，或对 read/write interface 加入结构约束。

这比“正交 mixer 更有表达力”更加准确，也更有理论深度。

---

# 六、用 controllability / observability 补上功能闭环

忽略特征维度，先看 stream-space 的线性化系统：

[
C_{k+1}=B_kC_k+\beta_ku_k,
]

[
r_k=\alpha_k^\top C_k.
]

其中 (u_k) 是 branch 写入，(r_k) 是 branch 能读到的 complement 内容。

令

[
\Phi_{j,i}
\==========

B_{j-1}\cdots B_i.
]

定义 stream-space controllability Gramian：

[
W_c
===

\sum_{k=0}^{K-1}
\Phi_{K,k+1}
\beta_k\beta_k^\top
\Phi_{K,k+1}^\top,
\tag{14}
]

以及 observability Gramian：

[
W_o
===

\sum_{k=0}^{K-1}
\Phi_{k,0}^\top
\alpha_k\alpha_k^\top
\Phi_{k,0}.
\tag{15}
]

如果

[
\lambda_{\min}(W_c)=0,
]

则某些 complement modes 从未被写入；如果

[
\lambda_{\min}(W_o)=0,
]

则某些 modes 从未被读出。

这给出了 identity-HC 与 learned IsoHC 的真正区别：

* identity-HC 中 (B_k=I)，若所有 (\alpha_k,\beta_k) 长期集中在相近方向，只使用了一个 complement mode；
* learned orthogonal (B_k) 可以不断旋转 rank-one read/write directions，使累计 (W_c,W_o) 达到满秩并改善条件数；
* 但如果 (\lambda_a,\lambda_b) 太小，或者所有方向高度共线，再好的 isometry 也只是保存“死状态”。

因此下一版理论主线建议变成：

> **Mean preservation 保住 residual identity；isometric transport 防止 passive information deletion；controllability 和 observability 决定 extra streams 是否真正构成有效容量。**

这也和 2026 年的 stream-collapse 工作形成了自然联系：该工作发现多 stream HC 中 residual mixing 往往接近 identity，信号和可解释特征集中于主导 stream，显式打破初始交换对称性可以改善利用率。你们当前已经通过 zero-mean `stream_embed` 和随机 read/write vectors 打破了最简单的复制对称性，但仍需直接度量 stream-space controllability 和 observability。([arXiv][7])

---

# 七、问题 1：最适合现代大模型的 HC 应是什么

## 7.1 不能简单把所有约束求交

你们论文已经正确证明：

[
\mathcal B_n\cap O(n)
\=====================

{\text{permutation matrices}}.
]

也就是非负、双随机、精确正交三者同时成立时，只剩离散 permutation。([GitHub][3])

因此不可能同时无代价得到：

* 连续可学习的非负平均；
* 精确等距；
* 完整连续表达力。

已有方法的核心 trade-off 是：

| 方法                  | transport 性质                         | 优点                       | 缺口                       |
| ------------------- | ------------------------------------ | ------------------------ | ------------------------ |
| Residual            | (H=I)                                | 最稳定、最便宜                  | 只有单 stream               |
| HC                  | unconstrained                        | 动态拓扑能力强                  | 谱不受控                     |
| mHC                 | Birkhoff                             | 保均值、非扩张、可融合              | complement 必然趋向扩散        |
| sHC                 | mean-preserving spectral upper bound | signed interaction、无非负限制 | 没有 (\sigma_{\min}) 下界    |
| JPmHC/orthogonal HC | orthogonal/Stiefel                   | skip 全谱稳定                | 未必固定 residual mean；无受控遗忘 |
| 当前 IsoHC            | fixed-vector orthogonal              | 精确保均值和 complement norm   | 静态、低自由度、可用性未保证           |

sHC 已经覆盖了 signed、mean-preserving、spectral-norm-constrained HC；JPmHC 已经直接提出 Stiefel/Cayley orthogonal HC 和 Jacobian-spectrum 分析；E-MHC-Geo 又进一步提出 input-adaptive Cayley orthogonal transport 和 reflection 分支。([arXiv][8])

所以你们下一步不能再把 novelty 写成：

> “首次用 orthogonal matrix 替代 Birkhoff matrix。”

真正有空间的方向是下面这个统一族。

---

## 7.2 建议方案：depth-budgeted bi-Lipschitz HC

所有同时满足

[
H\mathbf1=\mathbf1,
\qquad
\mathbf1^\top H=\mathbf1^\top
]

的矩阵，都可以写成

[
H=P+UBU^\top,
\tag{16}
]

其中 (B\in\mathbb R^{(n-1)\times(n-1)})。

对可逆 (B) 做 polar decomposition：

[
B=Q\exp(S),
]

其中

[
Q\in O(n-1),
\qquad
S=S^\top.
]

因此定义

[
\boxed{
H_k
===

P+
UQ_k\exp(S_k)U^\top.
}
\tag{17}
]

约束

[
-\kappa_k I
\preceq
S_k
\preceq
\kappa_k I.
\tag{18}
]

则单层 complement singular values 满足

[
e^{-\kappa_k}
\le
\sigma_i(B_k)
\le
e^{\kappa_k}.
\tag{19}
]

对任意深度复合：

[
\sigma_{\max}(B_{K-1}\cdots B_0)
\le
\exp\left(\sum_k\kappa_k\right),
]

[
\sigma_{\min}(B_{K-1}\cdots B_0)
\ge
\exp\left(-\sum_k\kappa_k\right).
\tag{20}
]

若施加全局 depth budget

[
\sum_k\kappa_k\le K_0,
]

则整个深度的 complement condition number 有统一上界

[
\kappa(B_{K:0})
\le
e^{2K_0},
\tag{21}
]

而不会随层数指数恶化。

这个结构自然地博采众长：

* (S_k=0)：就是当前 IsoHC；
* (S_k\preceq0)：允许有界去噪和遗忘；
* (Q_k)：提供 signed、无损 stream communication；
* (S_k) indefinite：允许受控放大与抑制；
* 它具有完整的 ((n-1)^2) 个自由度，覆盖所有可逆的 mean-preserving complement operators；
* 不要求非负，也不需要 Sinkhorn；
* 同时给出 (\sigma_{\max}) 和 (\sigma_{\min}) 的双边界。

纯 IsoHC 的连续自由度只有

[
\dim O(n-1)
\===========

\frac{(n-1)(n-2)}2.
]

当 (n=4) 时只有 3 个自由度；而完整 mean-preserving affine space 有

[
(n-1)^2=9
]

个自由度。式 (17) 正好补齐这 9 个自由度。

需要注意：这与 sHC 的 signed spectral 思路相邻。因此论文的新颖性必须落在：

1. 双边 singular-value bound，而不是只有 operator-norm upper bound；
2. 全局 depth budget；
3. full-block Jacobian theorem；
4. controllability/observability；
5. static gauge equivalence；
6. 大模型功能和系统验证。

---

## 7.3 完整 block 的统一稳定性条件

对式 (17)，令

[
\epsilon_k
\==========

\frac{|a_k||b_k|}{n}|J_k|.
]

若进一步允许 dynamic (H_k(X))，定义

[
\rho_k
======

\sup_{|\Delta X|=1}
\left|
DH_k(X)[\Delta X]X
\right|.
]

则完整 block 可得：

[
\boxed{
\sigma_{\min}(D\Phi_k)
\ge
e^{-\kappa_k}-\epsilon_k-\rho_k,
}
\tag{22}
]

[
\boxed{
\sigma_{\max}(D\Phi_k)
\le
e^{\kappa_k}+\epsilon_k+\rho_k.
}
\tag{23}
]

这给出了从 static IsoHC 走向现代 dynamic HC 的正确理论路线：

* transport radial budget：(\kappa_k)；
* branch Jacobian budget：(\epsilon_k)；
* router sensitivity budget：(\rho_k)。

仅仅证明每个输入点上

[
H_k(X)^\top H_k(X)=I
]

并不足以证明动态映射

[
X\mapsto H_k(X)X
]

是等距的，因为 (\rho_k) 仍可能很大。

---

# 八、当前理论完整度的具体评价

## 已经比较完整的部分

你们目前论文中以下部分是正确而且值得保留的：

1. mean direction 与 mean-zero complement 的分离；
2. Birkhoff non-expansive 但非 isometric；
3. (H_\rho) 的指数收缩反例；
4. fixed-vector orthogonal subgroup

[
\mathcal M_{\mathbf1}
\=====================

{Q\in O(n):Q\mathbf1=\mathbf1};
]

5. 非负、正交、stochastic 只能退化为 permutation；
6. fixed-vector polar projection；
7. 明确不声称 full-block dynamical isometry 或决定性 PPL 优势。

这些内容构成了一篇合格的 transport-mechanism note。([GitHub][3])

## 要达到强主会理论包，还需要补六组结果

### 1. Birkhoff 内点严格收缩定理

加入式 (7)–(8)，将“can contract”强化成：

> 只要 Sinkhorn matrix 保持严格正且离 permutation boundary 有统一距离，complement transport 就必然指数收缩。

### 2. 完整 mean/complement block Jacobian

加入式 (10)，明确 transport、read、write 和 branch Jacobian 的关系。

### 3. Static gauge-equivalence theorem

说明静态可逆 mixer 在一维深度链上可以被吸收到 read/write 坐标系中；当前方法的优势是 condition/optimization，而非无条件增加函数类。

### 4. Controllability/observability theorem

至少给出 (W_c,W_o) 的满秩充分条件，并证明 learned rotations 在什么条件下能够改善 Gramian condition number。

### 5. Bi-Lipschitz depth-budget theorem

把 IsoHC 放到式 (17) 的 (\kappa=0) 特例中，使理论不局限于“永不遗忘”这一极端。

### 6. Newton–Schulz 近似误差定理

当前 NS 对每个 singular value 的迭代为

[
s_{t+1}
\=======

\frac12s_t(3-s_t^2).
]

令

[
e_t=1-s_t^2,
]

则

[
e_{t+1}
\=======

\frac14e_t^2(3+e_t).
\tag{24}
]

这说明满秩、缩放后 (0<s_t\le1) 时二次收敛；但若 (s_t=0)，则永远保持 0。因此应明确：

* polar minimizer 在 complement block 非奇异时唯一；
* rank-deficient 时 projection 非唯一；
* finite-step NS 的误差依赖最小 singular value；
* 固定 (K=5) 不能对所有 (n) 和训练状态给统一保证。

---

# 九、代码和实验实现中应立即修正的地方

## 9.1 `mHC` baseline 命名与忠实度

`lm/mixing.py` 里的 `MHCMixing` 是静态 (n\times n) logits，加 10 次 Sinkhorn；官方 mHC 则计算 input-dependent dynamic mappings，并包含 static mappings和 gating。当前 baseline 适合叫 `static-birkhoff-hc`，然后另外实现 faithful dynamic mHC。([GitHub][5])

## 9.2 暴露初始化谱

至少暴露：

* `diag_bias`
* `temperature`
* `noise_std`
* `sinkhorn_iters`
* target distance to identity

并在训练前、训练中、训练后都记录 composite spectrum。

当前最重要的对照不是再增加一个新方法，而是：

[
\text{frozen-init Birkhoff},
]

[
\text{trained Birkhoff},
]

[
\text{same distance-to-identity controls},
]

[
\alpha Q\text{ contractive orthogonal controls}.
]

## 9.3 有限 Sinkhorn 的 complement leakage

当前实现最后执行 column normalization，因此 column-sum 误差很小，但 row-sum 误差约为 (0.0046)。这意味着实际矩阵不完全满足

[
H\mathbf1=\mathbf1,
]

从而 uniform mean state 可能泄漏到 complement：

[
U^\top H\mathbf1\neq0.
]

需要额外报告

[
\ell_{\mathrm{mean}\to\perp}
\============================

|U^\top Hv|_2
]

和

[
\ell_{\perp\to\mathrm{mean}}
\============================

|v^\top HU|_2.
]

最好加入一个 exact Birkhoff 参数化对照，如 TBP、go-mHC 或 mHC-lite；这些工作已经分别从精确性、完整表达力和参数效率角度改进了 Sinkhorn。([arXiv][9])

## 9.4 IsoHC 参数化应改成 minimal representation

当前 `H_raw` 有 (n^2) 个参数，但只使用

[
U^\top H_{\mathrm{raw}}U
]

的 polar factor。当 (n=4) 时，目标流形只有 3 个连续自由度，却用了 16 个 raw 参数，存在大量无效或径向方向。

建议主实现改为：

[
Q=P+URU^\top,
]

其中 (R\in O(n-1)) 直接用：

* Givens rotations；
* Cayley + 单独 reflection；
* Householder products。

对 (n=4)，(O(3)) 只需 3 个 rotation angles。这样可以：

* 精确正交；
* 不做 NS iteration；
* 避免 SVD fallback；
* 更容易 fuse；
* 避免 raw-polar 过参数化。

Cayley 只能覆盖不含 (-1) eigenvalue 的部分，因此若声称覆盖整个 (O(n-1))，需要 reflection/Householder component；并行工作已经专门讨论过这一问题。([arXiv][10])

## 9.5 统一 dtype 描述

`lm/mixing.py` 的 docstring 说 projection 在 fp32 中完成，但 `isohc/projection.py` 实际将小矩阵转为 float64，再转换回调用方 dtype；主 LM 又因 `torch.compile` 下 SVD 路径崩溃而关闭 fallback。论文和代码应统一描述为实际实现，而不是继续写 `bf16_fp32_mix`。([GitHub][11])

## 9.6 修正诊断指标

当前 `mean-zero energy` 实际计算的是

[
\frac{|P_\perp X|_F}{|X|_F},
]

它是 norm ratio，不是 energy ratio；若称 energy，应平方。当前 stream cosine 又是在完整 (X) 上计算，极易被大幅度 mean mode 支配。([GitHub][12])

建议改为：

[
r_{\perp}
\=========

\frac{|P_\perp X|_F}{|X|_F},
]

[
E_{\perp}
\=========

\frac{|P_\perp X|_F^2}{|X|_F^2},
]

以及：

* centered stream cosine；
* complement covariance effective rank；
* complement CKA；
* singular entropy；
* dominant-stream share；
* (\lambda_{\min}(W_c),\lambda_{\min}(W_o))；
* Gramian effective rank；
* mean/complement gradient norms分解。

当前 mHC diagnostics 还会过滤小于 (10^{-6}) 的 singular values，可能恰好隐藏真正塌缩的 complement direction。应始终在固定 (U) 基下计算恰好 (n-1) 个 singular values。

---

# 十、问题 2：下一步实验计划

## P0：先排除当前最致命的因果混淆

这是投稿前必须完成的最低优先级实验包。

### A. 初始化与训练贡献分离

对 current static Birkhoff baseline 扫描：

[
\texttt{diag_bias}
\in
{2,4,6,8},
]

以及多组 temperature。

每组记录：

* initialization single-layer (\sigma_\perp)；
* initialization composite gain；
* checkpoint composite gain；
* final composite gain；
* loss、gradient 和 complement use。

同时增加：

1. frozen (H)，只训练网络；
2. trainable (H)；
3. (H=\rho S+(1-\rho)I) 的 identity-annealed Birkhoff；
4. (B=\alpha Q) 的 signed contractive control；
5. IsoHC 人为乘 (\alpha<1) 的 contraction control。

这样才能区分：

* 非负性效应；
* contraction 效应；
* initialization 效应；
* learned routing 效应。

### B. 增加 faithful dynamic mHC

至少实现：

* dynamic read；
* dynamic write；
* dynamic residual mixer；
* static biases；
* learnable gates；
* 官方非负投影；
* 与官方相同的 Sinkhorn iteration 数和初始化。

否则主要结论必须严格限定为“static Birkhoff-HC”。

### C. 深度复合曲线

建议保持参数量大致固定，测试：

[
L\in{24,48,72,96,128},
]

注意实际 transport 数是 (2L)。

主图应画：

[
\log\sigma_{\max}(B_{2L:0}),
\quad
\log\sigma_{\min}(B_{2L:0})
]

对 (L) 的关系，并与

[
\sum_k\log\sigma(B_k)
]

的理论预测比较。

---

## P1：建立“保留 complement → 使用 complement”的功能桥

### A. 持续 clamp，而不是单点 removal

当前单层删除 complement 后，后续 branch 可以重新生成 complement，所以干预较弱。

使用：

[
X_l\leftarrow
\mathbf1\mu_l^\top,
\qquad
\forall l\ge k,
]

以及

[
X_l^{(\gamma)}
\==============

\mathbf1\mu_l^\top+\gamma P_\perp X_l,
\qquad
\gamma\in{0,0.25,0.5,0.75,1}.
]

预期应观察到随 (\gamma) 单调变化的：

* token-level NLL；
* KL divergence；
* sequence accuracy；
* copy/recall accuracy。

这也是你们论文 future-work 中已经提出的正确方向。([GitHub][3])

### B. read/write gate 因果扫描

至少比较：

[
\lambda_a,\lambda_b
\in
{0,0.01,0.03,0.1}.
]

更好的实现是每层、每分支单独使用有界 gate：

[
\lambda_{a,k}
\=============

\lambda_{\max}\tanh\theta_{a,k},
]

[
\lambda_{b,k}
\=============

\lambda_{\max}\tanh\theta_{b,k}.
]

必须报告：

* 初始值；
* 训练轨迹；
* 最终每层分布；
* (|\alpha_k|)、(|\beta_k|)；
* (|\alpha_k||\beta_k|)；
* (W_c,W_o) 的 spectrum。

### C. 专门设计必须使用 complement 的任务

推荐三个机制任务。

#### Multi-register delayed copy

在 (n-1) 个 complement modes 中写入不同变量，经过很深网络后分别查询。mean path 容量不足以无冲突保存全部变量。

#### Rotating read/write routing

每层只允许 rank-one write 和 rank-one read。只有通过 learned rotation 将不同方向循环搬运，累计 controllability/observability Gramian 才能满秩。

该任务最适合证明：

[
\text{learned IsoHC}>\text{identity-HC}.
]

#### Controlled denoising

信号与噪声同时进入 complement：

* pure IsoHC 应保留两者；
* Birkhoff 应一起耗散；
* (Q\exp(S)) 应能选择性压低噪声方向。

这个任务用于证明 pure isometry 并不是终点，受控 radial component 有必要。

---

## P2：有统计意义的 LM 实验

当前 48-layer 运行约 177M 参数，但只训练 20M tokens，即

[
T/P\approx0.113.
]

它非常适合快速机制检查，但不足以判断 learned complement 是否会在正常预训练后形成实际用途。当前主实验也只有一个 seed。([GitHub][4])

2026 年 stream-collapse 工作已经在约 120M 和 360M 模型上采用了更长 token budget，并报告 (T/P=10,20,40) 的 scaling；其 120M 主运行使用约 1.3B tokens。这个尺度应当作为当前 HC 方向更现实的比较参照。([arXiv][7])

建议最小实验矩阵：

| 规模        |               深度 |   Token budget | Seeds | 目的                           |
| --------- | ---------------: | -------------: | ----: | ---------------------------- |
| 100M–150M |         24/48/96 | (T/P=10,20,40) |     5 | 完整消融和统计显著性                   |
| 300M–400M |         24/48/72 | (T/P\approx20) |     3 | 验证 scale 与 depth interaction |
| 1B        | 标准深度 + deep-thin | 10B–20B tokens |   1–2 | 大模型确认                        |
| 3B，可选     |           一个确认配置 |    30B+ tokens |     1 | 强化现代 LLM 相关性                 |

主基线应包括：

* standard residual；
* identity-HC；
* static Birkhoff-HC；
* faithful dynamic mHC；
* sHC；
* JPmHC 或等价 orthogonal baseline；
* current IsoHC；
* proposed budgeted BiLip-HC。

`unconstrained HC` 可以保留为 instability oracle，但不能因为其短预算 loss 最低，就作为主要性能上界；当前 48L 结果中它确实 loss 最低，却明显违反 fixed-vector 和 orthogonality 约束。([GitHub][4])

所有结论至少同时画：

[
\text{loss vs tokens},
]

[
\text{loss vs training FLOPs},
]

[
\text{loss vs wall-clock},
]

[
\text{time-to-target-loss}.
]

下游可增加：

* HellaSwag；
* PIQA；
* ARC；
* WinoGrande；
* associative recall；
* multi-query copy；
* 长距离 retrieval。

对 ACL 类定位，还必须把 complement 的作用与语言现象联系起来；对 ICML/NeurIPS 类定位，则理论、机制和 scaling 可以作为主线。2026 年 CCF 第七版人工智能 A 类会议包含 AAAI、NeurIPS、ACL、CVPR、ICCV、ICML 和 IJCAI。([ccf.org.cn][13])

---

## P3：系统实验必须进入正文

当前 48L 报告中：

* identity-HC：34,351 tok/s；
* IsoHC：22,498 tok/s；
* mHC：16,846 tok/s。

因此 IsoHC 当前只有 identity-HC 约 65.5% 的吞吐，即约 34.5% slowdown。([GitHub][4])

而官方 mHC 在 kernel fusion、recompute 和通信优化后报告约 6.7% 的训练额外开销，这会成为审稿时的直接参照。([arXiv][2])

建议：

* 用 minimal Givens/Householder 代替 raw matrix + NS；
* 将 stream read、rotation、write、residual merge 融合成一个 Triton/TileLang kernel；
* 不显式 materialize (n\times n) matrix；
* backward 中重算小 mixer；
* 对 TP/PP 报告通信开销；
* 报告 HBM bytes、MFU、peak memory、kernel 数量和 multi-GPU scaling。

对 (n=4)，complement 只有 3 维，3 个 Givens angles 足够表示 (SO(3))。系统上没有理由让小型 stream-axis rotation 产生三成以上长期开销。

---

# 十一、当前论文的 novelty 应如何重新定位

截至 2026 年 7 月，正交 HC、Cayley/Stiefel、signed spectral HC、input-adaptive orthogonal residual 和 stream-collapse 都已经有直接相关工作。你们当前 bibliography 已经引用 sHC 和 KromHC，但还没有覆盖 JPmHC、E-MHC-Geo、stream-collapse、go-mHC 和 TBP-mHC。([GitHub][3])

因此建议将论文主张从：

> Birkhoff 不等距，因此我们提出 fixed-vector orthogonal HC。

升级为：

> **多流 residual connection 是一个具有固定 barycenter 的受控状态空间。Mean preservation 只保证标准 residual path；bi-Lipschitz transport 决定额外状态是否被动丢失；controllability 与 observability 决定它是否构成真实模型容量。**

对应的贡献可以组织成：

1. **Canonical decomposition**：mean identity + complement state-space；
2. **Strict Birkhoff contraction theorem**：Birkhoff 内点必然耗散 complement；
3. **Static gauge theorem**：静态可逆 transport 的函数类等价与条件数问题；
4. **Controllability/observability theory**：解释何时 rotation 真正优于 identity；
5. **Depth-budgeted bi-Lipschitz HC**：IsoHC 是 (\kappa=0) 特例；
6. **Functional and scalable validation**：从 synthetic necessity 到 100M/350M/1B LM；
7. **Minimal fused implementation**。

标题也可以从当前偏 mechanism-note 的标题，推进到类似：

> **Beyond Mean Preservation: Isometric and Controllable Residual Transport for Hyper-Connections**

或：

> **Hyper-Connections as Controlled Residual State-Space Transport**

---

# 十二、最终判断

当前仓库最强的地方是：

* 数学主问题抓得准；
* 论文没有夸大 PPL；
* 48L 单层谱与 96 层 composite gain 的数值闭环非常漂亮；
* identity-HC、complement removal 和 replacement controls 都已经意识到；
* GNN 实验也发现了“打破 stream symmetry、不要只做 mean readout”的关键架构问题。([GitHub][14])

当前阻碍 CCF-A 的四个核心 blocker 是：

1. **baseline fidelity**：当前 `mHC` 不是 faithful dynamic mHC；
2. **initialization confound**：`diag_bias=4` 已预置强 contraction；
3. **functional accessibility**：complement 被保存但 read/write coupling 太弱；
4. **novelty collision**：纯 orthogonal HC 已不够新。

最值得推进的不是放弃 IsoHC，而是把它重新定位为：

[
\boxed{
\text{mean-preserving bi-Lipschitz HC 中的 conservative limit}
}
]

并把整篇论文的理论核心提升为：

[
\boxed{
\text{Invariant mean}
+
\text{well-conditioned transport}
+
\text{controllability}
+
\text{observability}.
}
]

最优先的三件事依次是：实现并重命名忠实基线、完成初始化谱匹配与 depth scaling、加入 gauge/controllability 理论及必须使用 complement 的 routing 任务。完成这三步之后，再投入长 token budget 的 100M–1B LM 训练，论文的主张才会从“发现并修复 transport pathology”升级为“提出一种确实增加可用 residual capacity 的现代 HC 架构”。

[1]: https://github.com/misaya-yang/iso_HC/blob/main/lm/models.py "https://github.com/misaya-yang/iso_HC/blob/main/lm/models.py"
[2]: https://arxiv.org/html/2512.24880v2 "https://arxiv.org/html/2512.24880v2"
[3]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/paper/isohc_arxiv/main.tex "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/paper/isohc_arxiv/main.tex"
[4]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/0525_fe48_overnight_report.md "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/0525_fe48_overnight_report.md"
[5]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/lm/mixing.py "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/lm/mixing.py"
[6]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/0525_posthoc_eval16_report.md "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/0525_posthoc_eval16_report.md"
[7]: https://arxiv.org/html/2606.03483v1 "https://arxiv.org/html/2606.03483v1"
[8]: https://arxiv.org/abs/2602.18308 "https://arxiv.org/abs/2602.18308"
[9]: https://arxiv.org/html/2605.21724v1 "https://arxiv.org/html/2605.21724v1"
[10]: https://arxiv.org/html/2605.06729v1 "https://arxiv.org/html/2605.06729v1"
[11]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/isohc/projection.py "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/isohc/projection.py"
[12]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/lm/diagnostics.py "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/lm/diagnostics.py"
[13]: https://www.ccf.org.cn/Academic_Evaluation/By_category/ "https://www.ccf.org.cn/Academic_Evaluation/By_category/"
[14]: https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/comprehensive_summary.md "https://raw.githubusercontent.com/misaya-yang/iso_HC/main/docs/0605_alldoc/comprehensive_summary.md"
