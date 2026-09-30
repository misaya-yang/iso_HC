# 残差架构前沿、竞争方案与新颖性边界

核查：2026-09-29，R5。本轮分段伴随读写候选已停止推进，直接先行性与新颖性边界见本页R5节；R4及2026-09-25/26的RDM评估保留为先前设计背景，RDM已撤销主线资格。以下依据原论文页面、全文或原始出版记录；不是穷尽检索，也不是新颖性保证。检索不到同名结果不证明首次发现。

## 已经被覆盖的部分

| 原始来源 | 本轮核验范围与内容 | 对本项目的约束 |
| --- | --- | --- |
| [Hyper-Connections, 2409.19606](https://arxiv.org/abs/2409.19606) | 原始论文条目：多 residual streams 与可学习连接 | 多流、读写和 mixing 本身不是贡献 |
| [mHC, 2512.24880](https://arxiv.org/abs/2512.24880) | 原始摘要：约束 residual connections 并做规模化稳定/效率优化 | 必须面对实际动态架构，旧静态 proxy 不代表 faithful mHC |
| [sHC, 2603.20896](https://arxiv.org/abs/2603.20896) | 原始摘要/条目：Birkhoff 之外的谱约束 HC | signed/spectral mixer 已有邻近方案；本轮没有证明 IsoHC 与其完整参数化严格相同 |
| [JPmHC, 2602.18308](https://arxiv.org/abs/2602.18308) | 原始摘要：正交 HC、Jacobian 谱分析与投影实现 | 正交性及“改善梯度条件”的宽泛主张已有直接先行工作 |
| [oHC, 2609.02672v1](https://arxiv.org/html/2609.02672v1) | 全文 §3–6、附录14：均值/差分分解、全正交 HC、四元数构造、LM 对照 | 不再以补空间收缩或全正交 mixing 作为独立主创新 |
| [Attention Residuals, 2603.15031](https://arxiv.org/pdf/2603.15031) | 全文 §6.2、式(10)：HC 的深度核与 m-semiseparable 结构，状态扩展解释，identity transport 讨论 | “HC 是深度 SSM”“跨切分 rank≤streams”及其直觉不能作为首次提出 |
| [LAuReL, 2411.07501](https://arxiv.org/html/2411.07501v1) | 原始全文：低成本残差重加权及扩展 | 强单状态 residual-gain 控制必须进入主比较 |
| [Ablate and Rescue, 2603.14833v1](https://arxiv.org/html/2603.14833v1) | 全文 §3.4：冻结 routing、删除 streams、缓存激活恢复，用输出 KL 分析冗余和非对称性 | “流删除—恢复”不是新方法；需超出 checkpoint 依赖分析，做到可部署降阶、重训与预测 |
| [How Does mHC Use Its Residual Streams?, 2609.05309v1](https://arxiv.org/html/2609.05309v1) | 全文 §4.5：生产 checkpoint 的位置、token-mean 与路由干预 | mixer 的静态结构可能重要；观察接近 I 不等于直接删除无损 |
| [Ho–Kalman 1966 原始出版记录](https://ntrs.nasa.gov/citations/19670049337) | 状态空间实现的经典来源；本轮仅核验书目信息 | realization 与 Hankel 思想归属经典系统理论 |
| [Dewilde–van der Veen: time-varying computational networks](https://sps.ewi.tudelft.nl/pubs/spie92.pdf) | 作者机构托管论文，搜索索引可读到 time-varying cross-cut realization；全文打开失败 | 时变切分秩/最小实现也不是新发现；正式稿引用前补齐原文核读与书目信息 |

## 对附件的四点必要纠正

**oHC 的任务收益与交换机制不能直接画等号。** 其训练比较支持该设置下 oHC 优于 iHC/mHC，但附录14关闭交换后的主要指标是 stream cosine；SO(3) 保均值训练臂缺失。需要训练对照与任务端点才可建立更强归因。§5 明确冻结的是 `alpha_res`，并非附件所说的静态 bias；不能据此断言静态成分全被冻结。[原文](https://arxiv.org/html/2609.02672v1)

**静态结构可能功能重要，但不反驳 gauge。** 2609.05309 的早层 mixer 替换为 I，C4 PPL 增加约 41.4%；换成该层诊断集 token 均值只增加约 0.2%。这支持该 checkpoint 的层特定结构有作用。两种操作都不是同时变换读写与出口的 gauge 变换，更不是同低成本类的重训练比较。[原文 §4.5](https://arxiv.org/html/2609.05309v1)

**状态扩展与秩已经有明确先行工作。** AttnRes 不仅提出深度 duality，还写出 HC 核、半可分结构，并联系 identity-HC 与 state expansion。项目必须在可预测的压缩/保留规则及实际收益上产生新增内容，不能只换成“gauge/Hankel”术语。[原文 §6.2](https://arxiv.org/pdf/2603.15031)

**干预与恢复也需要区别于先行工作。** Ablate and Rescue 已用冻结 routing 的流删除和缓存恢复分析功能冗余。本项目候选应区别为：以受限输入—输出合同给出预测，构造真实的低阶递推，在 routing 重算和从头重训练中检验，并报告真实成本；目前尚未完成这一贡献。[原文 §3.4](https://arxiv.org/html/2603.14833v1)

## 本轮补齐的直接竞争与更早先行性

| 原始来源 | 核验范围 | 对 RDM 的直接约束 |
| --- | --- | --- |
| [RMT，ICML 2025](https://proceedings.mlr.press/v267/mak25a.html) | 正式会议条目/摘要：矩阵残差记忆，状态规模与主干计算宽度分离 | “从单流变矩阵记忆”不能再作为新增贡献 |
| [DDL v4，2601.00417](https://arxiv.org/abs/2601.00417v4) | 当前版本摘要，另读v3全文：读出、比较目标、rank-1纠正与改写 | read–compute–rewrite口号和有门的覆盖公式已有；不夸大单流无法学习内容替换 |
| [xHC v1，2607.14530](https://arxiv.org/html/2607.14530v1) | 全文§3.2–3.3、§4.5：丰富写回、稀疏更新、稠密读取 | 增加状态或只写少数槽位都不是新；需要比较同容量的写入丰富度与控制成本 |
| [Stream Collapse，2606.03483v1](https://arxiv.org/html/2606.03483v1) | 全文§2–3：dominant stream、Learned Stream Scaling | “多流未充分利用”“破对称初始化”已有直接工作 |
| [DNC，Nature 2016](https://www.nature.com/articles/nature20101) | 原始摘要与[作者团队说明](https://deepmind.google/blog/differentiable-neural-computers/)：内容访问、分配、释放 | 内存生命周期思想也不是首次出现；RDM必须胜过同骨架usage/free策略 |
| [Improving DNC，1904.10278](https://arxiv.org/abs/1904.10278) | 原始摘要：键值分离、释放aliasing、地址链退化 | key/value/version一致性是必要工程合同，不是可忽略实现细节 |
| [DeltaNet 作者说明](https://sustcsonglin.github.io/blog/2024/deltanet-1/) | 作者机制解析：关联记忆的键干扰与有限容量 | “键碰撞/只写误差/遗忘”已有系统认识；正式稿应补引对应原论文 |

上述方案在不同数据、规模和实现下报告结果，不能把论文表中的数字拼成统一SOTA榜单。RDM尚未训练，不进入任何胜负排序。

## 已撤销RDM提案的区别设想与缺口

RDM的具体区别设想是：**因果预测后续深度读取需求，用保留期限约束提交与释放，在固定容量下管理表示的存活。** 当前主实现使用独立槽位、归一化读取与有界凸替换。它不试图发明slots、控制器、凸更新或cache eviction。

| 最接近者 | 需要新增的可审查内容 | 不能算新增 |
| --- | --- | --- |
| DNC / cache管理 | 后续depth需求监督与因果预测，相对usage/FIFO/free的额外价值，成熟Transformer训练与成本证据 | 加一个分配/释放门 |
| DDL / DeltaNet | 新写入不得侵害仍需存活的其他内容，容量满时显式准入，保护与释放的共同价值 | 加法变rank-1覆盖 |
| xHC | 保留期限与提交语义，而非仅稀疏选流；同写回丰富度和状态量的比较 | n变大或少写几个流 |
| AttnRes | 相近检索能力下固定容量的需求驱动保留，优于固定分块/FIFO的质量—成本 | 将过去层attention改为较短窗口 |
| mHC / oHC | 绕开任意全流transport，直接检验内容存活与修订的工作流；真实训练稳定与收益 | 另一种等距或谱约束 |

新颖性检验分三层：单项原语多数已知；这套depth工作流的具体组合尚需持续检索；组合是否值得采用只能由匹配消融和强baseline训练证明。搜索暂未命中同名方法不构成首次性证据。

## 工程约束与贡献判断（R3）

2026-09-26重新核读工程章节：AttnRes通过分块、两阶段读取与融合控制数据搬运；xHC-Flash通过复用跨子层读取/路由降低流量。这些事实支持“系统映射必须与算法共同设计”，不证明所有复杂架构无效，也不构成对未跑过的RDM的实测慢速结论。[AttnRes §4](https://arxiv.org/pdf/2603.15031)、[xHC](https://arxiv.org/html/2607.14530v1)。

RDM同时引入调度、硬决策和额外监督，还改变了标准残差的恒等路径；目前没有不可替代收益来抵付这些复杂度。因此撤销其优先级。这是R3的决策依据；当前R4已实现的最小更新见 [架构选择](architecture.md)。

## 对“solid accept”的实质判断

Gauge、Hankel阶、WD归因、训练过的SO(n−1)对照，可以构成研究基础或附属结果；当前不足以单独承担所要求的贡献度。上一版“能压缩一些streams”同样不够。

如果RDM最终只等同DNC式缓存，或只有值归一化有用，应按实际结果重归因。值得主会强主张的结果应让读者获得一个此前缺少的架构选择：**有限深度工作区可以按后续需求保留并安全复用，而且在成熟LLM上比直接累加、全部混合或固定历史分块更值得采用。** 这是RDM当时的假说，当前未建立且不再主导研发。整体目标仍是能改变架构选择的质量—成本收益。

此前RDM检索涉及learned eviction、future-use预测和slot allocation，保留为历史范围。当前专项比较转向tied read/write、子空间project/lift、dynamic identity-HC、初始化及单流残差归一化；不将已撤销路线的检索清单当成研发任务。

## R4伴随读写算法的直接先行性

当前算法详见 [architecture.md](architecture.md)。一般HC的自由读写包含这一受限模型类，RMT已有检索与外积写回；不能以“同一表示空间读写”宣称首次。[HC](https://arxiv.org/abs/2409.19606)、[RMT](https://proceedings.mlr.press/v267/mak25a.html)。

[Chimera §3.6](https://arxiv.org/html/2607.28611)在视觉扩散模型中采用动态读写、H_res=I的iHC，因此取消mixer也不是新增概念。我们的待测区别是单位地址与严格adjoint绑定、其局部条件数合同和baseline-exact可学习初始化；它是否改善实际LM尚未验证。

[CliffSearch作者论文](https://cliffsearch.ai/assets/cliffsearch_preprint.pdf)的导出资产含GrassmannianSubspaceRouting：子空间投影读、lifting写及额外门。它是必须比较的机制邻近；其Table6将raw-best G3/H2判为跨样本泄漏无效，另一个同名节点H1通过特定审计，不能把节点混为一个有效SOTA结果。当前实现没有复用其代码，并在完整模型上单独测试token因果性和样本隔离。公式邻近不因某节点实现错误而消失，首次性仍未建立。

[Orthogonal Residual Updates，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/67c15da4a9340140c60783d9a175fd3f-Abstract-Conference.html)将更新在特征维上正交化；本算法在stream维选择读写地址，不应混称为同一个“正交残差”方法。仍需在真正任务上比较相关强方案，而非只靠概念区别。

## R5新增原文核查与被排除的改名路线（2026-09-29）

本轮原文读取由主代理执行；子代理提供独立本地代数反驳，没有将子代理报告误记为网页全文核验。检索不是穷尽性保证。新候选 phase-adjoint 的规则见 [architecture §8](architecture.md)，证明见 [theory §11](theory.md)。

| 原始来源与本轮范围 | 对当前设计的实际约束 |
| --- | --- |
| [AttnRes原PDF，§3–4、§6.2](https://arxiv.org/pdf/2603.15031) | 已有固定pseudoquery、Block历史、两阶段读取及online-softmax合并。固定状态/深度核解释、可训query和融合不能独立作为新增贡献；不能拿其naive实现证明新方法更快。 |
| [SANA-Video2.0，§3.3，式(2)](https://arxiv.org/html/2607.21553v1) | 已跨深度共享attention与FFN query，源含completed blocks和可变partial。共享查询已有直接方案；将固定query的统计流式化首先是执行方式，反复提交变化的partial会错误计数。该工作在视频扩散，不等于LM同资源结果。 |
| [MHAR，原全文与摘要](https://arxiv.org/html/2607.27230v1) | 已按feature子空间拆深度softmax、多头及保函数delta转换。head split和zero-gated转换不作为新原语。本轮没有独立复现其训练/production kernels。 |
| [Momentum ResNets，ICML2021正式条目](https://proceedings.mlr.press/v139/sander21a.html) | 通过动量改造残差已是已发表方向；residual+EMA/二阶更新不因改叫depth memory成为新方法。这里只核验正式条目和摘要，未把全部LM后续文献穷尽。 |
| [Set Transformer，ICML2019正式条目](https://proceedings.mlr.press/v97/lee19d.html) | learned seed pooling和固定潜在摘要有更早邻近；shared content pooling也必须与有限特征/因子化attention区别。 |
| [DDL v4全文，§2–5](https://arxiv.org/html/2601.00417v4) | 同方向read-compare-write、scalar与expanded state已有直接构造；其单run、equal-token及容量归因边界按原文保留，不拼为统一SOTA榜单。 |
| [xHC全文，§3、§5](https://arxiv.org/html/2607.14530v1) | 丰富写入和跨子层读取复用已有算法/工程方案。新方法需算反向与数据流，不能仅按新增参数或FLOPs推断快。 |

共享bank的代数若为 (y=\sum_r a_rU_r/Z_r\)、(U_r=\sum_j e^{p_r^Tk_j}v_j\)，则是有限特征归一化attention；混合pool不等于以混合query作softmax。[Efficient Attention](https://arxiv.org/abs/1812.01243)、[LambdaNetworks](https://arxiv.org/abs/2102.08602)、[Agent Attention](https://arxiv.org/abs/2312.08874)是子代理提出的直接公式邻近，**本轮尚未逐式核读其全文**。因此bank不作为本轮新主方法。其固定prefix检索span/rank不超过bank数，加入uniform方向最多再加1；这只是接口限制，不是完整LM能力下界。

phase-adjoint也不是逃出先行性压力：全局是identity-HC的互逆尺度aligned read/write，局部是一次固定scaled mixer加阶段内unit tied。待测区别是exact-baseline、真实跨cut历史的一阶信号和局部对齐的组合，最强替代是gated boundary skip；普通gauge、初始化或局部条件数本身不承担首次性。若这些已有机制的简单控制恢复全部收益，按实际结果重归因。


末端反例进一步收窄新颖性：`terminal-adjoint`在全部body之后做一次出口混合，就打开全部body writer的真实history梯度，因此“首次打开残差历史信用分配”或“必须内部阶段”均不可由当前证明支持。内部phase相对terminal真正待测的是**更早让历史改变后续branch计算是否值得**；terminal在非零路由后也会读取辅助状态，不能将它描述为训练期间始终仅做late pooling。两者都保留两条流和同数量router，较少内部结构不等于已测更便宜。

§11.8的AR/Brownian Gram刻画承担指定函数保持接口的设计预算；其经典分解、正尺度/串行/精确条件以及近似与并列reader反例必须保留。论文若推进，应把贡献落在可用初始化、窄接口刻画和强对照之外的架构选择，不能将参考地址秩改叫任务记忆容量下界。

已进一步核读 [CliffSearch节点公开源码](https://cliffsearch.ai/best-node/) `g002_n0024_66b987`：读 \(U\alpha\)，写 \(\operatorname{diag}(\beta)U\gamma\)，一般不单位绑定，也不固定自作用为1；逐流门可将write移出所投影子空间。具体限制、初始化与一次交汇的区别见 [theory §10.4](theory.md)。该核查只建立直接project/lift先行性和合同区别，不借node alias或自动review评分认定benchmark有效、工程更快或创新成立。

本轮关闭后，不再以这些数学/参数化区别维持phase优先级。早期双LR前缀未超过强gain，尚无支撑主会方法主张的额外质量—成本证据；见 [收尾报告](reports/R5_CLOSEOUT_20260929.md)。
