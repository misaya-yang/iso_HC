# 实验路线与结项决定：R5 已关闭

状态：**R5 CLOSED，按用户要求本轮结项。** 当前没有活动R5训练优先级或自动续训安排。算法和代码保留为研究资产；结果、实际训练范围、费用及关机记录由 [R5 closeout 报告](reports/R5_CLOSEOUT_20260929.md) 统一管理。R4作为参考资产，RDM保持归档。

## 当前关闭决定

已完成的gain/phase两组LR前缀均为每臂33,554,432 tokens。3e-4下gain/phase持出NLL为 `5.387013 / 5.391525`；6e-4下为 `5.065748 / 5.086086`，两组均由gain领先。完整更新profile中phase比baseline慢约18%，gain接近baseline。terminal仅有3e-4完整前缀，与phase相差不超过0.000208 nat；其6e-4轨迹中断，不能据此宣称跨LR确认或内部交汇等价。

用户已要求关闭本轮，原实例已于23:51 UTC确认OFF。停止R5优先推进，原续训、补齐七臂、增seed和扩规模安排均失去执行效力。268,435,456 tokens的完整诊断未完成；本轮结论限定为短预算质量—成本尚未支持继续优先投入phase，不是成熟LM方法族失败的结论。后续研究另设明确问题、依据和授权。

## 原计划参考（CLOSED）

以下D0–D1、七臂/D预算、归因规则和扩规模安排保留为封存计划，均不构成当前执行指令。**完整七臂前缀加268,435,456 tokens续训协议为 NOT_EXECUTED / CLOSED**；已经完成的准备、profile和部分前缀只按closeout的实际回执登记，不能把计划表当完成清单。

## 0. 原问题与实验摘要（已封存）

已有事实：R4完整模型和tiny合成NTP可训练，但收益与冻结输入carrier几乎重合，且真实历史核初始一阶导数为0。R5的一次固定深度交汇让真实早层branch获得一阶梯度，并保留阶段内固定地址的局部伴随合同。**同一梯度可由gated boundary skip得到，且仅末端交汇已能打开全部body历史梯度**，所以首轮必须面对terminal、boundary-skip和post-frozen。

主终点是固定持出文档的token加权NLL随真实训练tokens和总时间的曲线。完整训练吞吐/峰值显存、route变化、pre/post辅助写入和任务影响用于归因；谱、写入量和可训合同不代替质量。若gain、boundary skip或Block AttnRes在质量—成本上已经更值得采用，就改变设计决定，不扩展控制器来维护故事。

## 1. D0：原准备合同（已封存，完成状态见closeout）

- 独立核验初始化因果核、真正history的一阶导数、坐标/全局函数与梯度等价、局部谱和多阶段反例。回执入口：[verify_phase_initialization.py](../../experiments/verify_phase_initialization.py)。
- 完整模型核验共同主干/初始logits/主干梯度、prehistory梯度、非零路由下token因果及batch隔离、BF16有限性；主方法与strongskip/post-frozen初始梯度对齐。
- 新runner验证全局update/LR/token预算、固定非重叠validation、缓存不存在/全NaN失败、独立data RNG、完整optimizer-boundary保存、连续训练与resume一致。旧runner保留作历史入口，不用于R5科学排序。
- 原始数据先按文档内容hash分流train/validation，再tokenize；相同文档不得跨split。记录官方dataset revision、tokenizer、EOS、cache身份和实际token数，不将随机tokens当语言数据。

9月29日执行时用户曾授权AutoDL准备与后续带卡启动；这段是当时授权背景，本轮已按用户要求关闭，不作为新租、开机或继续训练的授权。具体endpoint、实例身份及OFF确认见closeout，不把历史 `agent.md` 当活机器记录。

## 2. D1：原单卡诊断与七臂协议（完整协议未执行，已关闭）

首个形状建议24L×256d、4heads、T512、GPT-2词表50257，约31.88M参数；所有方法共享主干seed、数据顺序、有效batch、branch初始化缩放、optimizer分组和评价集合。默认中间block为唯一交汇点，第一轮不搜索最优cut。使用标准NTP，不增加teacher或辅助loss。

| 第一组训练臂 | 科学职责 |
| --- | --- |
| 调好的 `baseline` | 真实优化水平与成本参照 |
| `gain` | 标准增益/尺度是否已解释改善 |
| `phase-adjoint` | 候选 |
| `terminal-adjoint` | 仅在出口交汇的更小替代，检验内部交汇是否必要 |
| `boundary-skip` | 同初始history梯度的强最小替代 |
| `phase-adjoint-post-frozen` | 持续post记忆更新是否必要 |
| `block-attnres` | 相近深度聚合的强方法，不拖到最后才比较 |

`phase-adjoint-shear`和R4在初始profile中一起测；若候选显示有意义信号，shear尽早进入匹配训练来检验对齐，而不将旧四流静态proxy冒充自由动态对照。`phase-adjoint-frozen`只是死区/入口负控制，不作为取得论文收益的弱对手。

推荐开发观察范围为约250–500M实际tokens/臂，保存中途checkpoint和曲线；该范围是小规模机制开发，不是充分成熟LM/SOTA。预算按GPU实测吞吐调整。短段只用于发现故障和估计学习状态，不凭任意几分钟null宣判方法族失败。若baseline仍未学会数据分布、评价没有区分力或运行中断，标未判定并修定位问题。

强baseline先采用所有臂一致的输出投影depth-scaled init；LR从已有合理配方的小范围选择，如3e-4/6e-4，以开发集选择后冻结。不能只给候选调参，也不能机械相同batch使较快/省显存的baseline浪费资源。科学比较共同有效batch，真实系统比较另允许各臂合理microbatch并单列其身份。

原拟完整协议（**NOT_EXECUTED / CLOSED**）为：七臂各做LR 3e-4/6e-4的2,048 updates前缀（33,554,432 tokens），从开始即使用总16,384 updates的同一全局日程、256 updates warmup及0.1倍末端LR。按**前缀最后checkpoint**的全持出集NLL选择LR，相等选3e-4，失败轨迹不参与。每臂选中前缀精确resume到16,384 updates（268,435,456 tokens）；末端较优也不能删掉其内部phase对照。所有臂model seed419、data seed20260929、microbatch32/accum1，固定3,906个validation blocks、1,999,872 targets，每2,048 updates评价/保存。该seed是开发，不是独立确认。实际只完成closeout登记的部分前缀，没有完成此七臂与完整D预算。

若boundary/post-frozen各自选中LR与phase不同，在七条主要轨迹后，以phase选中LR继续对应已训练前缀，post-frozen优先于boundary。主质量比较用各自相同调参预算，机制差值用匹配LR；优化控制恢复收益时，不宣称机制必要。保留自动关机前30分钟的保存余量，并以实测吞吐逐阶段检查能否在UTC07:15结束；预算不够的轨迹明确记未完成，不能用不同tokens排序。第一次GPU成本上限按20元控制，provider UTC07:45关机，不自动追加预算。

核心开发seed与确认seed分开；初步配对重复覆盖方法效应与训练随机性，不把文档bootstrap当训练seed。必要的同width浅深对照回答方法×深度互动，不能将改变width后的近参数匹配直接解释为纯深度效应。

## 3. 原工程测量与有限算力预算（已封存）

先对每个实际shape/臂测完整forward、CE、backward、AdamW和zero_grad后的稳态tokens/s；CUDA同步、编译预热、eval/checkpoint分别记录。显存探测必须有Adam状态，不能只forward/backward；用共同effective batch和梯度累积匹配优化更新。每个验证点记录跨resume的累计时钟；checkpoint须连同timing sidecar保存。first-update/warmup可能包含lazy compile与allocator启动，不能把wrapper构建时间称纯编译时间。

若实测吞吐为 \(r_i\) tokens/s，臂i的训练tokens为 \(D_i\)，seed数为 \(s_i\)，则训练小时估算为

\[
H_{train}=\sum_i s_iD_i/(3600r_i).
\]

另外加上实际compile、validation、checkpoint与准备时间。不引用旧5090回执预测新vGPU速度，不先给虚构的GPU小时保证。先测再配置有效实验；无卡准备阶段不让付费GPU空等。局部kernel快不能冒充端到端快。

本版主干是GPT-2式GELU和绝对位置编码；结果只覆盖该诊断骨架。出现可复核收益后，应转到现代RoPE/SwiGLU主干验证兼容性。当前没有KV cache，暂不主张decode延迟、KV增量或分布式通信收益。

CUDA初始化核验已发现默认Inductor no-grad评价对phase与baseline产生不同数值路径，旧默认编译one-step的约0.005 nat差不计作质量收益。正式训练显式使用`emulate_precision_casts=True`，反向在autocast之外时指定`backward_pass_autocast=off`；所有持出NLL使用原始模型的共同eager BF16评价。数值策略写入resume身份。更改策略后重新初始化质量轨迹，不继续旧默认策略的preflight checkpoint；旧测量原样保留。

## 4. 原机制归因与决定规则（已封存）

首轮主终点是在事先冻结的实际预算 \(D\) 后，**最后 checkpoint** 的固定持出集token加权NLL；`best_nll`仅记录训练过程，不用于方法排序。开发阶段的投入优先阈值预设为 \(\delta_q=0.02\) nat（约2% perplexity），不是方法有效性的普适边界；小于该值或单seed不确定的结果仍可按成本和后续精度决定，不能当作等价证明。预算由实测吞吐和明确费用上限确定，在查看正式比较的质量结果前冻结。

关键差值为 \(L_{boundary}(D)-L_{phase}(D)\) 和 \(L_{post-frozen}(D)-L_{phase}(D)\)。归因必须按阶梯展开：phase对post-frozen主要检验post持续写入；post-frozen对boundary-skip检验pre读取/绑定增益这一组合，不能将其直接解释为对齐收益；对齐解释需要匹配shear训练。phase只胜过boundary-skip属于复合差异，不能独立证明post记忆必要。

开发展示固定tokens及累计时间的完整曲线；若报告达到共同NLL的时间，目标须在相应确认比较前冻结，未达到明确记未达到。训练seed提供训练随机性的证据；持出文档重采样只度量评价不确定性，连续checkpoint不当作独立重复。预算不足时收窄此次结论。确认阶段为各强方法提供相同调参预算，不能将共用一个开发LR直接称为各方法充分调优。

| 观察 | 下一决定 |
| --- | --- |
| terminal匹配或优于内部phase | 内部交汇未显示必要性，优先末端构造；共同收益不全归于内部phase |
| phase超过gain，但boundary skip同样好 | 优先更简单skip；撤销“阶段内可更新记忆必要”主张 |
| post-frozen同样好 | post持续写入未显示必要性；保留pre交汇机制的可能收益 |
| shear更好 | 对齐约束可能损失有效表达力；不把局部定理当训练最优 |
| candidate质量更好但慢 | 用达到共同目标NLL的总成本判断；只优化已定位的工程瓶颈 |
| route/谱改善而任务无变化 | 不升级为方法优势；检查具体错误，否则结束该代理诊断 |
| 候选收益在充分开发段消失 | 收窄到短期优化效应；不把早期点估计写为成熟收益 |
| 有重复的质量—成本优势且强消融支持 | 方法冻结，进入独立尺度/seed确认 |
| 实现故障或训练未有效完成 | 未判定；恢复有效checkpoint并修明确错误 |

初始化gradient匹配是归因控制，不是所有臂训练轨迹都应该相同。临时checkpoint ablation只说明当前模型依赖；从头训练的strongskip/post-frozen才检验新增机制是否值得。

## 5. 原2027主会贡献与扩规模合同（未启动，已封存）

论文应形成一个连贯结论：**残差历史可以在精确baseline起点获得有效信用分配，而阶段内绑定读写在限制表达力后仍更值得采用。** 需同时具备最近先行工作的逐式区别、有效自然语言训练、强skip/gain/shear/AttnRes比较、至少两个相关尺度/深度条件、关键确认的独立训练seed和实测质量—成本。定理、测试、单seed短训练都不能独立承担该主张。

开发通过后，100–150M约2–3B tokens用于稳定比较和配方选择；方法冻结后，350M等独立量级、关键至少3个确认seed及未用于开发的持出文档。约1B/20–30B tokens是资源允许且贡献需要时的扩展，不是本轮默认launch或录用的普适必要条件。若只能获得小模型结果，论文主张须相应收窄，不能靠SOTA措辞填补规模证据。

若最终只留下一个更好的HC初始化/skip，应重新判断贡献度，不能强行沿用“新的残差记忆工作流”。共享query bank因直接对应factorized attention和SANA/MHAR不在本轮并行开发；RDM、压缩优先和旧IsoHC谱故事都不恢复。
