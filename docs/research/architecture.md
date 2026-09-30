# 算法研究资产：已关闭的 R5 与 R4 参考实现

状态：**R5 CLOSED，按用户要求本轮结项。** `phase-adjoint` 不再具有活动研究优先级；原续训、补齐对照与扩规模安排不再构成执行指令。本文保留全部算法合同、实现和强对照作为研究资产；实际结果与关机记录统一见 [R5 closeout 报告](reports/R5_CLOSEOUT_20260929.md)。

关闭依据：在每臂33,554,432 tokens的已完成前缀中，gain在两个LR上均有更低持出NLL：3e-4为 `5.387013 < 5.391525`，6e-4为 `5.065748 < 5.086086`（gain < phase）。完整更新profile中phase比baseline慢约18%，gain接近baseline。terminal只完成3e-4前缀，与phase相差不超过0.000208 nat；6e-4轨迹中断，不能跨预算排序。268,435,456 tokens的完整诊断未完成，这些结果支持本轮停止优先推进，不构成成熟LM方法失败的结论。

R4参考记录（2026-09-26）：**已实现完整Transformer模型与训练入口，并保留数学合同、tiny合成训练和CPU成本回执；尚无成熟LM性能或SOTA结论。** 主实现为 [lm/adjoint.py](../../lm/adjoint.py)。以下第1–7节保留R4合同，第8节保留已关闭的R5构造；待验证问题不自动触发新训练。RDM保持归档。

## 1. 一个操作：在哪里读，就沿同一单位方向写回

每个token维护两条d维流，X=[x;m]。每个attention/MLP子层选择单位地址c=(c₀,c₁)，执行

\[
z=c_0x+c_1m,\qquad \delta=F(\operatorname{Norm}(z)),
\]
\[
\boxed{x^+=x+c_0\delta,\qquad m^+=m+c_1\delta.}
\tag{A1}
\]

读取和写回使用**同一个**c，不另学一个write gate，也不施加H_res矩阵。两条旧状态的carry是identity。attention与MLP仍各调用一次，与原模型主干的宽度、参数和顺序一致。

动机是消除读写错位产生的额外剪切，而不是要求所有状态永久等距。分支可以沿被访问方向学习增、减、覆盖；没有被该步读取的正交方向不被该次写入改变。

## 2. 地址如何生成

实现中每个子层只有一个逐token标量线性路由器：

\[
u=b+w^\top\operatorname{RMSNorm}_0(x)/\sqrt d,\quad t=\tanh u,
\qquad c=\frac{(1,t)}{\sqrt{1+t^2}}.
\tag{A2}
\]

RMSNorm₀没有可学习缩放。动态版本每个路由器d+1个参数；静态对照只保留b。attention、MLP、最终出口各有独立路由器。L层的新增参数为(2L+1)(d+1)，没有新增embedding矩阵、teacher、额外目标或离散决策。

t限制在[-1,1]，因此当前实现的角度范围为[-π/4,π/4]；它不是任意SO(2)旋转或完整自由读写。静态地址下，写到读的核为c_tᵀc_j，是单位对角Gram核的因果部分；在当前角度范围它非负。这个限制可能有利于优化，也可能损害能力，须与自由读写控制比较。

地址在FP32构造（float64输入保留float64），随后以同一转换规则用于read/write。完整BF16模型保持流状态BF16；有限精度下cᵀc≈1而不是精确1。没有NS/SVD/Sinkhorn、全流矩阵求逆或状态裁剪。

## 3. 三条可核验性质

对一次已选定的单位c，记r=(-c₁,c₀)，G=F∘Norm：

\[
c^\top X^+=c^\top X+G(c^\top X),\qquad
r^\top X^+=r^\top X.
\tag{A3}
\]

也就是被访问的坐标执行普通residual，其正交坐标保持。若分支输出为零，则两个状态都严格保持；这对输入依赖c也成立，不需要先把地址变回identity。

固定地址时，在[c,r]坐标中的完整子层Jacobian为

\[
\operatorname{diag}(I+J_G,I).
\tag{A4}
\]

更强的比较是：保持同一个读取c、同一个G，并要求cᵀb=1，其他write可写成b=c+v、v⊥c。此时产生额外的vJ_G剪切；伴随写回b=c同时最小化最大奇异值、最大化最小奇异值，因此在非奇异时最小化局部2-范数条件数。证明见 [theory.md §10](theory.md)。这是固定读取合同下的比较，不是所有架构中的全局最优性。

动态c(X)的真实Jacobian还含路由导数；不能由(A4)宣称整网梯度稳定，也不能由本地数值测试推断训练必胜。

## 4. 精确baseline初始化，同时避免路由完全学不动

所有路由器w=b=0，故c=(1,0)。入口为

\[
x_0=E_{tok}+E_{pos},\qquad m_0=D x_0,
\tag{A5}
\]

D是固定交替±1的对角符号矩阵，不训练、不增加参数。最终出口也初始化为c_out=(1,0)。这时每个branch输入、输出和最终logits都与共享权重的baseline相同；没有norm cap或破坏恒等路径的预混合。

为什么辅助流不初始化成零？全零辅助流且所有地址对齐时，角度的一阶读写贡献都不可见，路由梯度为零。Dx₀提供初始不被读出的、通常非共线的载体：当x₀在两种符号组均有非零分量时，它与x₀非共线。角度变化能通过branch读取形成一阶任务梯度。简单复制x₀在第一步主要是径向变化，可能被RMSNorm抑制；固定符号变换避免这一常见退化，但不保证对所有样本/参数梯度都非零。

此构造只让**入口作用**产生一阶路由梯度。全对齐时深度核c_tᵀc_j的角度导数仍为零，不能声称已经让所有记忆路由方向一阶可学习。若branch的局部导数为零，也可能暂时无路由信号。

另一个实质混杂：Dx₀可能只是有用的embedding长skip。因此实现了冻结辅助写入对照；它保留同样载体与读取能力，只禁止把后续branch输出写进m。若该对照保留全部收益，就不能归因于可更新多流记忆。

## 5. 实际实现与控制

| 方法ID | 实现区别 | 用途 |
| --- | --- | --- |
| `adjoint-hc` | 动态单位地址，同地址写回，signed carrier | R4参考候选与原训练资产 |
| `adjoint-hc-static` | 只学习每层标量地址 | 判断token条件化是否必要 |
| `adjoint-hc-frozen-aux` | 辅助流固定为Dx₀ | 判断收益是否只是输入长skip；此控制不满足完整伴随写回 |
| `adjoint-hc-zero-carrier` | 辅助入口为零 | 对齐初始化死区的负控制 |
| `adjoint-hc-copy-carrier` | 辅助入口复制x₀ | 区分非共线载体与普通复制 |

实现复用 `BaselineTransformer` 的注意力、MLP、归一化、embedding/head及state_dict键。路由器零初始化不消费随机数，同seed主干一致。 [训练工厂](../../experiments/lm_5090_next_runs.py)明确登记n_streams=2；旧四流配置不会被静默套用。

原训练器的AdamW分组未在本次修改，旧入口仍可能对所有参数做WD；正式公平比较须显式记录并一致处理各臂的参数分组。当前不是优化过的推荐大规模训练配方。

## 6. 工程账与现实限制

新增运行状态是一条d维辅助流。每个子层额外操作是逐元素读写、RMS归约、长度d的单输出投影及每token标量归一化，量级O(d)；没有K² transport、条件索引或生命周期元数据。

O(d)不等于没有开销：参考PyTorch实现增加kernel/中间张量，辅助流也增加读写及保存激活。CPU完整forward/backward测量只说明本机实现，不能外推GPU吞吐；GPU融合、BF16与activation checkpoint仍需实际测量。不能把参数增加很少当作内存或延迟增加很少。

## 7. 先行性与论文潜力

这是HC/RMT允许类中的一个受约束设计；无H的动态iHC、子空间project/lift和破初始化对称都已有邻近工作。我们不声称首次提出多流、同子空间读写或固定sign种子。

可争取的贡献是：**以精确残差语义约束读写，消除可避免的错位剪切，并用可训练初始化把它变成一个足够便宜、有实际收益的实现。** 是否有独立贡献必须由最接近的自由读写/子空间对照及充分LM训练决定，不能靠局部定理单独获得SOTA资格。相关文献见 [literature.md](literature.md)。

代码、合同测试与集成probe是R4交付。以下保留R5设计依据与强对照合同；本轮已关闭，实际决定见 [closeout](reports/R5_CLOSEOUT_20260929.md)。

## 8. 已关闭的 R5 构造：一次深度交汇，打开真实历史的一阶学习路径

R4 在精确 baseline 起点的历史核仍为二阶；输入 carrier 的梯度与有用历史学习不能混称。R5从零辅助流开始，在每个阶段内部保持同一个单位地址读取和写回：

\[
m_0=0,\quad c_\ell=(1,\tanh u_\ell)/\sqrt{1+\tanh^2u_\ell},\quad
z=c_0h+c_1m,\quad \delta=F(\operatorname{Norm}(z)),\quad
(h,m)^+=(h,m)+(c_0\delta,c_1\delta).
\tag{P1}
\]

在预先确定的中间 Transformer block 入口只做一次

\[
\boxed{(h,m)\leftarrow(h+m,m-h).}
\tag{P2}
\]

所有 router 为零时，边界前 (m=0)，边界 active 不变、辅助变成负的边界快照，之后的 active 与 baseline 按相同顺序更新。初始函数与共同主干梯度因此匹配 baseline。早期 router 的辅助写入却在边界被 active 读取，其一阶信号是实际 branch 创新与边界伴随的内积，而非 signed 输入载体。

这一实现等价于全局坐标的互逆尺度同方向读写：读 \(c^TX/a\)，写 \(ac\delta\)，第一阶段 \(a=1\)，第二阶段 \(a=1/\sqrt2\)。它明确改变了 R4 的全深度单位系数合同。边界(P2)是一次固定的缩放旋转，奇异值为\(\sqrt2\)，不能宣称整个局部存储空间等距；阶段内固定地址的局部 Jacobian 合同仍成立。完整证明与“一次非共线切换”限制见 [theory §11](theory.md)。

### 8.1 能新增的能力与最强替代解释

新接口允许早期创新经第二流绕过若干后续分支，到一个深度交汇点影响后半网络；它同时允许阶段内使用不同视图。普通NTP直接训练，不保存所有层输出，不添加外部监督、槽位元数据或controller。

但一个 gated boundary skip 已能产生相同的一阶梯度。更小的末端构造也可让全部body writer的一阶历史梯度打开。第二阶段的初始辅助值还包含边界长skip；第一个post分支的读扰动可能接近径向，被RMSNorm抑制。因此**“梯度打开”不能作为内部交汇或可更新记忆必要性的结论**。需要证明更早的历史读取及阶段内写回比这些替代多出的作用值得采用。

### 8.2 方法与强对照的身份

| 方法ID | 更新差异 | 能回答的问题 |
| --- | --- | --- |
| `phase-adjoint` | (P1)+(P2)，零辅助入口、动态单位地址 | 已关闭的R5诊断候选 |
| `terminal-adjoint` | 相同零入口、body单位tied与output router；唯一(P2)在全部body之后 | 内部交汇是否必要；全部body writer在起点获得末端history梯度 |
| `boundary-skip` | pre分支只读h，h按baseline更新；零门将δ累加到m；相同边界交汇；post读取固定边界carrier，不再写m | 相同一阶history信用分配是否已足够；排除delayed skip解释 |
| `phase-adjoint-post-frozen` | pre与主方法相同，post禁止辅助写入 | 跨边界路径可训后，持续更新第二流是否必要 |
| `phase-adjoint-frozen` | 全程禁止辅助写入，仍执行边界转换 | 初始化/入口负控制；pre路由死区，不能作主要强基线 |
| `phase-adjoint-shear` | 同读取、同边界，写 (b=c+s(-c_1,c_0))，s零初始化 | 对齐约束是否值得；保持 (c^Tb=1)，明确属于自由写控制 |
| `gain` | 单流 (h^+=h+[1+\tanh u(h)]\delta)，零路由初始化 | 非零一阶导数的强单流尺度控制 |
| `block-attnres` | 每个目的子层的独立query、completed summaries与mutable partial、softmax读历史 | 相关强方法；参考执行程序不是原论文生产kernel或完整配方复现 |

R4 `adjoint`/frozen/shear 另作为比较臂，原 `adjoint-hc` 训练工厂和历史结果不覆盖。full-frozen属于负控制，不靠它的死区制造胜负。所有控制中的快照保留对早层主干的正常梯度，不暗中detach。

末端交汇必须在body循环结束后明确执行，不能只允许`boundary=L`却遗漏转换。terminal在非零路由后仍读取并更新辅助流，与纯weighted late skip仅初始一阶相同。最终router在零点主要产生径向扰动，可能被RMSNorm抑制；这不阻断body writer的history梯度，亦不保证所有新增参数都有强梯度。证明见 [theory §11.9](theory.md)。

### 8.3 实现、成本与部署合同

实现入口：[phase_adjoint.py](../../lm/phase_adjoint.py)、[Block AttnRes](../../lm/block_attnres.py)。统一诊断 runner：[residual_lm_diagnostic.py](../../experiments/residual_lm_diagnostic.py)。正式证据状态以 [evidence](evidence.md) 为准。

主方法维持两条 `[B,T,d]` 流、每个branch一个scalar router、每个attention/MLP一次主干调用。边界额外读取两流并产生两条新状态，发生一次；每子层仍有归约、读写和反向中间量。router使用R4同一FP32地址构造，流 dtype 按模型/AMP实际行为记录。参考实现无条件索引、Sinkhorn/SVD、逆矩阵或自定义调度。

不增加的只是 attention/MLP 调用次数；辅助流、router和训练保存/重算都有成本。原FP32参数+AMP路径的残差流可能仍为FP32，不能按BF16流估显存。尚无KV-cache实现，因此当前只有训练与prefill能实测，不能称全序列重复forward为decode benchmark。

该构造属于一般HC的受限参数化，不是从函数类上超越HC。合同成立与新增机制值得采用是不同结论；本轮结果未支持继续优先投入phase，故实现、初始化证明和强对照全部保留为研究资产。共享query bank、EMA和多时间尺度保留为分析材料，不自动启动新主线。RDM不恢复。
