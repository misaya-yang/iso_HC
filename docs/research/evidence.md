# 证据账本：已有结果、边界与缺口

更新日期：2026-09-26，R4。审计起点：`main@d4af4d6`。本页替代旧综合报告的**当前研究判断**，保留旧报告和原始文件不变；逐项来源和状态见 [evidence_registry.json](evidence_registry.json)。

## 1. 当前结论

**R4研究状态：`adjoint-hc` 已实现完整Transformer与训练入口，并完成四臂合成NTP集成训练；路由梯度和辅助写入已激活，当前没有有意义的质量收益证据。** 两个CPU形状的完整前向加反向耗时为baseline的1.417×与1.157×。精确结果和测量范围见第9节。RDM仍已撤销主线资格；旧slot原语验证不恢复其优先级。

仓库已有可靠的算子数值证据，也有真实 LM 和 GNN 的正、负结果。它们支持“等距 transport 能保留补空间”“多 stream 架构能缓解这里的深层 GNN 退化”，但尚不支持“IsoHC 有稳定的 LM 性能收益”“补空间能量就是有效记忆”或“正交交换已经被证实是收益来源”。

应保留的主资产是：有原始摘要的 LM 无明显收益结果、transport 与任务损失脱节的干预结果、GNN 架构与 mixing 的对照、实现与优化器审计。本轮还新增了独立的 gauge 数值验证、WD-only 反事实重放和 transport 诊断修复，见第 8 节。Claude 原始结果文件仍不在当前 checkout；新验证与附件所述原运行分别登记。

## 2. 证据等级

| 状态 | 含义 | 可以写什么 |
|---|---|---|
| `local_raw_checked` | 本轮读取历史 JSON/log，与报告核对 | “已核对本地历史记录”；不写“本轮重跑训练” |
| `local_code_checked` | 本轮核查当前源码 | 当前代码具有某结构；不自动证明旧远程运行使用完全相同代码 |
| `historical_report_only` | 只有历史 Markdown 汇总，没有核对逐次原始记录 | 保留其数值和不确定性，不重新计算显著性 |
| `attachment_reported_unverified` | Claude 附件声称完成，当前 checkout 缺脚本/原始产物 | “附件报告”；不能写“我们已复现” |
| `new_local_verification` | 本轮真正执行且留下命令/结果的离线验证 | 只覆盖实际验证对象，不提升为训练证据 |
| `local_integration_training` | 本轮实际优化完整模型，有配置、逐步记录和本地产物 | “已完成合成训练集成验证”；不提升为自然语言效果或规模化稳定性 |

9月25日的审计没有启动训练；它独立执行WD-only重放与两组gauge数值验证。9月26日R4新增了真实的本地合成训练，单独登记，不能倒填为旧实验或Claude原运行的复现。`results/gnn_stage1/` 中若干 JSON 是当前机器本地文件，未被 Git 跟踪；可供本轮核对，但克隆仓库不保证获得它们。下面保留可携带的关键数值，完整重算仍需归档这些原件。

## 3. 真实 LM：三组实验必须分开

### E-LM24：TinyStories，24 层，20M tokens

来源：[0524 报告](../0605_alldoc/0524_lm_experiment_report.md)及 `docs/0605_alldoc/0524_results_raw/0524_deep_stress_512/*/run_summary.json`。训练日志记录约 **101.52M 参数**；`d=512, h=8, context=512, streams=4, seed=0`，各臂实际训练 `20,004,864` tokens。AdamW：`max_lr=3e-4, min_lr=3e-5, warmup=2M tokens, weight_decay=0.1`。

| 方法 | batch | optimizer steps | val loss | PPL | 历史 tok/s |
|---|---:|---:|---:|---:|---:|
| baseline | 48 | 814 | 2.431165 | 11.372 | 153,891 |
| unconstrained | 12 | 3,256 | 2.196444 | 8.993 | 40,914 |
| mHC | 12 | 3,256 | 2.248664 | 9.475 | 30,633 |
| IsoHC | 12 | 3,256 | 2.260183 | 9.585 | 67,539 |

**可以得出**：本次 IsoHC 比 mHC 差 `0.011519` nats；unconstrained 在三个 HC 臂中最好。**不能得出**：相对 baseline 的差全部来自架构，因为 baseline 的 batch/更新次数不同，20 个 validation batch 对应的样本量也不同；没有 identity-HC 或多 seed。汇总文件 `deep-stress-512_summary.json` 只含补跑的 IsoHC，完整四臂需读各自 `run_summary.json`，不能把顶层文件当全量结果。

### E-LM48：FineWeb-Edu fair-batch，48 层，20M tokens

来源：[0525 overnight 报告](../0605_alldoc/0525_fe48_overnight_report.md)及 `docs/0605_alldoc/0525_results_raw/0525_fe_fair_deep48_p33013_20m/`。约 **177.04M 参数**，`d=512, h=8, context=512, streams=4, seed=0`。所有方法 `batch=20, steps=1,954`，实际 `20,008,960` tokens，验证 20 batches。数据为 `sample-10BT` 的本地 token cache，使用 GPT-2 tokenizer。相同 batch 改善了优化预算可比性，但不代表各方法都达到各自最佳吞吐。

| 方法 | val loss | PPL | 历史 tok/s | 相对 identity 吞吐 | 最终补空间 energy |
|---|---:|---:|---:|---:|---:|
| baseline | 5.736599 | 310.008 | 48,359 | 1.408 | — |
| identity-HC | 5.715724 | 303.604 | 34,351 | 1.000 | 0.05025 |
| unconstrained | 5.675997 | 291.779 | 26,333 | 0.767 | 0.05467 |
| mHC | 5.724873 | 306.394 | 16,846 | 0.490 | 0.01362 |
| IsoHC | 5.720640 | 305.100 | 22,498 | 0.655 | 0.05456 |

IsoHC 比 mHC 好 `0.004233`，却比 identity-HC 差 `0.004916`；unconstrained 比 identity-HC 好 `0.039728`。这是一 seed 的排序，没有跨 seed 置信区间。IsoHC 的历史吞吐是 identity-HC 的约 65.5%，是明确需解释的成本；不能把该编译/批量配置的速度当作优化后内核的固有上限。

最终 mHC 补空间单步奇异值均值约 `0.92221`；IsoHC 约 `0.99996`，固定向量误差约 `2.79e-7`，正交误差约 `5.57e-4`。mHC 的有限次 Sinkhorn 行误差约 `0.0046`、列误差约 `2.30e-7`：这是近似双随机实现，不能把精确双随机的块对角公式无误差地代入数值结果。unconstrained 的几何漂移是被测现象，不能仅因它违反拟议约束而从性能主表排除。

本 run 配置明确 `save_checkpoints=false`。它不能承担 checkpoint 干预的复现来源。

### E-POSTHOC48：另一个 48 层 checkpoint 的 eval16 分析

来源：[posthoc 报告](../0605_alldoc/0525_posthoc_eval16_report.md)及 [本地 JSON](../0605_alldoc/0525_results_raw/0525_core_mhc_isohc_48l_p51198/posthoc_eval16/mechanism_analysis_summary.json)。这是 `p51198` 训练来源，**不是**上节 `p33013` fair-batch run；两者 base loss 不同，不能合并比较。报告称 48 层、20M tokens；本地 posthoc JSON 可核对 seed0 checkpoint 名称和 **96 次 transport**，但未包含完整训练 config、训练吞吐或可加载的本地 checkpoint。本次 probe `batch=1, eval_batches=16, intervention_stride=4`。

| 方法 | base loss | 旧版 projected-step product SV 均值 | 去除补空间的最大 loss 增量 | 全部替换为 I 的增量 | 全部替换为 random Iso 的增量 |
|---|---:|---:|---:|---:|---:|
| mHC | 5.900727 | 0.000438950 | +0.000288814 | -0.000523716 | -0.000831664 |
| IsoHC | 5.901274 | 0.996349037 | +0.000516444 | +0.000712991 | +0.000559598 |

**字段语义纠正**：这些旧 JSON 没有 `schema_version`，按 legacy/schema-1 解读。旧 `composite_sv_*` 实际计算 `B_k…B_1`，其中 `B_i=UᵀH_iU`；它是在每一步投影后的乘积，不自动等于完整 transport 的 `Uᵀ(H_k…H_1)U`。各 H 保持补空间（`e₀ᵀH_iU=0`）已足以使两者相等，不要求完整双随机。旧 mHC 的列和接近精确，意味着此条件接近成立；其较大的行误差主要对应 mean→complement，**不能仅凭行误差断言旧 composite 明显错误**。历史字段仍按实际算子定义使用，对一般交换矩阵则必须区分两种乘积。schema-2 已同时报告两者，见第 8 节；旧 checkpoint 尚未据此重新分析。

这里最大值是在所测 intervention 位置中挑出的描述性统计，不能作为未校正的显著性证据；另有位置 removal 后 loss 下降。raw 记录的梯度 profile 斜率分别为 `-0.021109` 和 `-0.021373`，没有显示 IsoHC 的端到端梯度优势。即便完整 transport 等距，也不等于包含注意力、MLP、归一化和读写后的整体 Jacobian 等距。

**保留结论**：该 checkpoint 的补空间 transport 差异很大，但测得任务作用很小。**不保留结论**：“更大的补空间 energy 已带来更好的 LM”“mHC 的 contraction 是模型为任务主动学出的”“如此小的干预量证明所有 HC complement 都无用”。

## 4. 本轮源码核查：为什么原实验可能缺少辨别力

来源：[lm/models.py](../../lm/models.py)、[lm/mixing.py](../../lm/mixing.py)、[lm/train.py](../../lm/train.py)。

1. **静态 transport**。当前 `StreamMixing.forward()` 不接收 token/state；mHC 是每个分支独立的静态 logits 加 Sinkhorn，IsoHC 是静态矩阵投影。它们不是生产级、输入依赖 mHC/oHC 的 faithful 实现。
2. **读与写是两端耦合**。`TwoBranchHCTransformer` 每层有 attention、MLP 两次 transport，各分支各自有读写向量；但 `readout_lambda` 与 `injection_lambda` 分别跨全部层/分支共享，初值都为 `0.01`，final-readout 另有标量。向量为 `a=1+λ_a P⊥u`、`b=1+λ_b P⊥v`，读出还除以 stream 数 `n`。在**精确保均值**且 transport 不混合 mean/complement 时，某次写入到后续读取的 kernel 项为 `1 + (λ_a λ_b/n) u⊥ᵀ C v⊥`。因此被任务访问的写—传—读项初始有约 `1e-4` 的双端缩放，而不是只受一次 `0.01` 门控。实际权重范数、初始化 complement、有限 Sinkhorn 误差和训练中 λ 的变化仍需测量，不能把这个阶数直接当作已测 loss 大小。
3. **全参数 weight decay**。训练器对 `raw_model.parameters()` 建立同一个 AdamW，历史 config 的 `weight_decay=0.1`，没有专门的 mixing/gate/norm/bias 分组；因此 transport logits、读写向量和共享 λ 都进入 decay 组。本轮独立 WD-only 重放能解释记录中大部分均值收缩，见第 8 节；尚未证明任务梯度为零或逐层学习轨迹完全由 WD 决定。
4. **均值改变包含多种机制**。`H1≠1` 的范数不区分放大、衰减或 mean↔complement 旋转；unconstrained 更好也不自动意味着交换更好。需拆分主通道增益、跨子空间交换、读写自由度与动态性。

这里支持的是“原设计可能弱耦合且有优化器混杂”，不是先验判定动态 IsoHC 必然等于 identity。动态模型及受限读写类需要单独的理论与实验。

## 5. 算子与数值资产

### E-DETECTOR：1024 层 residual-only

[原始 JSON](../0605_alldoc/0510_results_raw/stage2_stability_detectors_core/stability_detectors.json)配置：`seed=123, trials=16, feature_dim=1024`，stream 数 `4/8/16`，深度 `32…1024`。下表是 `fp32, n=8, L=1024` 的均值。

| 方法 | raw energy ratio | mean-zero energy ratio | grad ratio | stream cosine |
|---|---:|---:|---:|---:|
| IsoHC | 0.999992 | 0.999991 | 0.999992 | 0.025019 |
| mHC-lite | 0.352515 | 3.52e-8 | 0.353616 | 1.000000 |
| gcn-diffusion | 0.356815 | 3.29e-8 | 0.353339 | 1.000000 |
| unconstrained | 5.51e13 | 5.61e13 | 5.58e13 | 0.987729 |

旧综合表把 mHC-lite 的 `energy` 写成 0，实际接近 0 的是 **mean-zero energy**，总 raw energy 不为 0。这里证明指定随机 transport 的乘积性质；没有任务训练，不代表所有 unconstrained 模型必然爆炸，也不证明含非线性网络在 1024 层训练稳定。“唯一稳定几何”不成立，identity 本身也是等距。

### E-PRECISION / E-PROJECTION

[精度 JSON](../0605_alldoc/0510_results_raw/stage2_precision/precision_depth_suite.json)中 `n=4,L=512`：`bf16_all` mean-error **均值** `0.538603`、**最大值** `0.695255`；`bf16_fp32_mix` 均值 `1.34e-5`、最大 `1.55e-5`。这是保留 fp32 mixing 路径的工程依据，不能把最大值 0.70 写成所有 trial 的平均误差。

[leftfix 投影 JSON](../0605_alldoc/0510_results_raw/stage1_cuda_projection_leftfix/projection_sanity.json)在 `n=4` 的 SVD / K=5 分别记录 mean orth-error `5.08e-7 / 5.07e-7`；旧报告另报 K=5 `1.80e-4`。它们不能不注明版本地合并。本次核对的 JSON 缺完整 fallback 配置，所以不据此宣称“纯五步 NS 在任意条件下均达 SVD 精度”；实际 LM 关闭 fallback 后误差约 `5e-4`，应按部署路径单独测。

## 6. GNN：架构收益明确，mixing 归因仍不充分

### E-GNN-SYNTH / E-GNN-CORA

本地原始记录 `results/gnn_stage1/synthetic_v2/oversmoothing.json` 与 `results/gnn_stage1/cora_isores_fast/cora_isores.json` 已核对，但不受 Git 跟踪。Synthetic SBM depth128：GCN 的 energy `0.200341`、graph-native v-centered variance `0.001212`；IsoNode 分别 `1.0/1.0`，cosine `0.032477`。全局保方差是算子性质，不能替代任务有效性。

Cora 单 seed42（seed/200 epochs/训练配置取自[历史报告](../0605_alldoc/gnn_experiment_report.md)）：

| 模型 | L2 accuracy | L16 accuracy | L32 accuracy |
|---|---:|---:|---:|
| GCN | 77.51% | 30.66% | 31.19% |
| ResGCN | 75.05% | 30.03% | 29.55% |
| IsoStream v2 | — | 69.10% | 63.93% |

IsoStream L16/L32 参数量分别 `159,552 / 225,344`；对应 GCN `157,696 / 223,232`。本地 `cora_ablation_sparse/cora_isores.json` 还核对了 v1→v2a（stream embedding）和 v1→v2b（concat）各自的大幅恢复，但 concat 改了读出参数化、dropout 可破坏对称性，不能把架构包收益全部归因于 IsoHC。

### E-GNN-MULTI-REPORT：五 seed 的历史汇总

`results/gnn_stage1/cora_multiseed/cora_multiseed.md` 保存 seeds `[0,1,2,3,4]` 的 mean±std；当前该目录未发现逐 seed JSON。本表为 **summary-only**，没有重算统计：

| 模型 | L2 | L16 | L32 |
|---|---:|---:|---:|
| GCN | 77.64±1.90% | 31.10±0.23% | 31.16±0.29% |
| ResGCN | 75.80±1.16% | 28.77±4.02% | 28.58±2.81% |
| IsoStream v2 | — | 73.11±1.88% | 66.64±3.42% |

### E-GNN-H：只改 H 的结果会改变旧叙事

单 seed42 的 `results/gnn_stage1/cora_h_ablation/cora_isores.json` 确认 orthogonal L16 `76.84%` 最高；但同实现的 identity 与 none 在 L32 为 `70.41% / 56.00%`，提示 RNG/初始化及训练噪声未良好配对。

另有 **五 seed 汇总** `results/gnn_stage1/cora_h_multiseed/cora_multiseed.md`（无逐 seed JSON，以下完整抄录为 Git 可携带证据）：

| H 类型 | L16 mean±std | L32 mean±std |
|---|---:|---:|
| identity | 74.05±1.70% | 66.44±0.89% |
| none | 74.18±1.05% | 69.04±1.93% |
| IsoHC | 73.95±2.17% | 70.68±3.07% |
| orthogonal | 72.14±1.66% | 68.96±5.80% |
| unconstrained | 68.59±2.86% | 62.52±4.60% |

这份较完整历史摘要**不支持**“打破均值约束的变体都更好”：orthogonal 的 L16 均值低于 IsoHC/identity。L32 IsoHC 比 identity 高 4.24 个百分点，但仅比 none 高 1.64 点；未恢复逐 seed 配对、初始化控制和不确定性分析前，不写“显著优于无 mixing”。这些结果共同支持：multi-stream/readout 架构有很大贡献，mixing 的增量需重新因果识别。

## 7. Claude 附件：保留假设，明确不可用资产

保真节选及来源状态见 [claude_handoff.md](claude_handoff.md)。附件称本地分支 `claude/wonderful-gauss-fsuyj9` 的提交 `5c047df` 含 `docs/0924_salvage/`，但本次审计起点未见该目录。以下数值仅是附件报告：

| 附件结果 | 报告数值 | 当前使用方式 |
|---|---|---|
| 静态 IsoHC gauge logits 检查 | float64 差约 `3e-16` | 原运行仍未恢复；第 8 节有新的独立验证 |
| WD-only 48L 单步谱 | 预测 `0.92174`，实测 `0.92221` | 原脚本/记录仍缺失；第 8 节有新的独立重放 |
| WD-only 24L 单步谱 | 预测 `0.91542`，实测 `0.91563` | 同上 |
| WD-only 48L composite | 预测 `4.0e-4`，实测 `4.39e-4` | 同上；相近不等于损失梯度严格为零 |
| CPU char-level probe | 约 0.8M 参数、16 层、2 seeds，seed 波动约 0.03 nats | 缺数据定义、完整配置、逐 seed raw 和 scaled-init 记录 |
| CPU 非等距 gain 收益 | unconstrained 约改善 0.05，自由 gain 约 0.03 | 假设：主路径增益可解释部分 HC 收益；待强单流对照 |
| CPU 强耦合 removal | loss 增加约 0.2–1.2 nats，但训练 loss 仅改善约 0.01 | “被使用”与“有相对收益”需分开；只是附件报告 |

附件的 Stage-1 通过概率 25–35%、预算 300–500 GPU 小时、两 seed 的“2σ”门槛是建议或主观判断，不是实验证据。本项目不把它们当作已测置信度、预先授权的 GPU 预算或适用于任意规模的终止定律。

## 8. 本轮新执行的离线验证

### E-NEW-GAUGE：真实模型与理论边界

来源：[复现说明](../../results/theory_audit_20260925/README.md)、[主回执](../../results/theory_audit_20260925/gauge_contracts.json)、[seed7 回执](../../results/theory_audit_20260925/gauge_contracts_seed7.json)、[独立验证脚本](../../experiments/verify_gauge_contracts.py)。环境为 Python 3.9.6 / PyTorch 2.8.0 / CPU / float64；两组各 **11/11 checks 通过**。`20260925` 与 `7` 是数值示例 seed，**不是两次训练 seed**。

真实 `TwoBranchHCTransformer` 经完整读、写、exit 坐标变换后，精确 float64 basis + SVD 投影的 logits 最大误差两组均为 `5.55e-17`；默认投影分别 `1.85e-9 / 5.50e-9`，与有限投影约束误差相容。只把 H 替换为 I 而不转换读写，会产生明显差异。因此静态 gauge 等价指参数变换后的函数等价，不是 checkpoint 可任意删去 mixer，也不是同一优化器轨迹等价。

反例检查同时限制理论外推：一般正交变换可能令 exit 离开仓库固定和的参数类；奇异 transport 不能由可逆坐标变换普遍化为 I；即便 `H=I`、不发生 mean↔complement 交换，合适的读写也能形成 rank-2 的跨深度 kernel。固定中心化读写向量的 kernel 修正按 `λ_a λ_b` 缩放，但不直接给出训练后的 loss 界。

### E-NEW-WD：独立 WD-only 反事实

脚本从各历史 run 读取 config 和实际 optimizer steps，使用当前 `cosine_lr_schedule`，按 `q=∏_t(1−η_t·WD)` 缩小对称 logits `4I`。四 stream 的补空间谱预测为 `(exp(4q)−1)/(exp(4q)+3)`。它省略初始化噪声和任务梯度，属于解析反事实重放，没有训练新模型。

| run | steps | 本轮 WD-only 单步谱预测 | 历史单步谱均值 | 绝对差 |
|---|---:|---:|---:|---:|
| FE48 fair-batch p33013 | 1,954 | 0.92178339 | 0.92220568 | 0.00042229 |
| TinyStories24 | 3,256 | 0.91555435 | 0.91562865 | 0.00007430 |

这支持“初始化加 WD 足以解释测得均值收缩的大部分”，不支持“任务梯度为零”。当前代码的 LR 日程也不构成旧远程 source identity 的完整证明。48 层的 96 次 transport 对称反事实预测 `0.00040214` **只是一项预测**，没有将它同另一个 p51198 checkpoint 的旧 posthoc 值当作匹配实测。

### E-NEW-TRANSPORT-SCHEMA：完整乘积与逐步投影乘积分开

[lm/transport_analysis.py](../../lm/transport_analysis.py) 的 `collect_transport_report` 已改为 schema 2：`composite_sv_*` 计算 `Uᵀ(H_k…H_1)U`，另加 `projected_step_product_sv_*` 保存 `(UᵀH_kU)…(UᵀH_1U)`。两次 90° mean–complement 旋转会令后者为 0、前者为 −I；旧诊断会遗漏经 mean 返回 complement 的路径。

本轮执行 `python3 -m unittest tests.test_transport_composition -v`，**3/3 通过**：覆盖交换后返回、保补空间算子的旧新一致性、非交换矩阵的前向乘法顺序。来源：[测试](../../tests/test_transport_composition.py)。这是诊断契约修复的验证，没有重新分析缺失的历史 checkpoint，也没有改变历史 JSON 的数值。

### E-RDM-CONTRACTS：已撤销候选的原语实现，非训练结果

[独立脚本](../../experiments/verify_depth_memory_contracts.py)、[解释](../../results/depth_memory_contracts_20260925/README.md)与[回执](../../results/depth_memory_contracts_20260925/contracts.json)记录CPU float64的 **11/11** 项验证。覆盖protected rows精确保持、invalid槽位按零内容写入、value/key/valid/expiry/version同步提交、拒绝持久写入但保留workspace、到期复用、深度前缀因果性和4,096步有界状态。原语直接接收当前proposal和外部给定expiry，没有学习控制器。

同时核验两个反例：有界状态仍可有大于1的完整导数；保住一个线性view的非正规更新仍可能放大。因而不能把这些检查升级成“RDM解决全部梯度不稳定”。64次写入由3个槽位承接的trace使用已知寿命，属于外部调度合同，不是模型学会了预测、压缩或推理。未来读取评分的选择界也以已知误差界为条件。

这项原语工作使候选变得可检查，**不支持C2机制成功或C3性能前沿**。本记录不再建议开发RDM学习控制器。若未来凭新依据重开方向，需重新建立必要性和工程合同；不得复用旧IsoHC loss作为RDM结果。

## 9. E-ADJOINT-INTEGRATION：完整算法、真实合成训练与成本

来源：[实现](../../lm/adjoint.py)、[合同测试](../../tests/test_adjoint_contracts.py)、[可重跑脚本](../../experiments/adjoint_hc_probe.py)、[完整回执](../../results/adjoint_hc_20260926/probe.json)与[解释](../../results/adjoint_hc_20260926/README.md)。方法定义见 [architecture.md](architecture.md)。以下是本轮实际执行，不是计划。

**实现核验。** 新方法16项合同测试通过，覆盖同seed主干/随机数一致性、初始logits完全一致、signed/zero/copy入口梯度、零分支恒等、固定地址的Jacobian、动态放大反例、非零路由下的token因果性和batch隔离、BF16状态、优化/存取、控制组及真实`run_single`训练入口。入口测试完成4次更新、评估、双流诊断和摘要落盘。旧训练入口14项测试也通过。它们不计为论文质量结果。

**真实集成训练。** 四臂共享初始化seed419和预生成训练batch；2层、d64、4头、词表32、上下文32、batch4。每臂80个AdamW步骤、10,240训练tokens，总计40,960；lr=0.001、WD=0.01、betas=(0.9,0.999)、梯度裁剪1。每次验证2,048 tokens。数据为一阶置换Markov链：0.8概率走固定后继，其余均匀随机；训练/验证独立采样。它不测试自然语言或长程记忆。

| 实验臂 | 参数量 | 初始val NLL | 最终val NLL | 学后单次前向的辅助写入RMS |
| --- | ---: | ---: | ---: | ---: |
| baseline | 102,720 | 3.465736 | 1.246839 | — |
| adjoint动态signed | 103,045 | 3.465736 | 1.246644 | 0.00504630 |
| adjoint静态signed | 102,725 | 3.465736 | 1.246806 | 0.00163801 |
| adjoint动态、冻结辅助写入 | 103,045 | 3.465736 | 1.246640 | 0 |

四臂初始logits最大差为0，全部loss/梯度有限。signed入口的初始路由梯度L2为0.0193373，zero入口为0；动态路由参数训练后L2移动0.283938。辅助写入量测的是同一前向内m_final−m_initial，排除了训练中embedding改变的混淆。**曲线几乎重合，冻结辅助写入同样有效；不把约0.0002的单次NLL差写成优势。** 这说明普通NTP可以激活路由和辅助写入，没有证明记忆更新必要，也不能凭此简单数据否定LM潜力。

**真实工程代价。** CPU单线程float32，交替测量完整forward+CE+backward+zero_grad，不含optimizer/data/transfer。2层d128/B2/T64/V32中位数：baseline 1.775646ms、adjoint 2.515833ms（1.417×）；4层d512/B2/T128/V128：29.997250ms、34.699125ms（1.157×）。均预热3次，分别测20/8次；逐次样本保存在回执中。两个形状都保留；深度、上下文和词表也变化，不能归因于单独增宽。参数增量很小不代表开销可忽略，CPU结果不预测GPU效率。

完整模型通过`torch.compile(backend="aot_eager", fullgraph=True)`前反向一致性，logits/参数梯度最大差均为0；这是图捕获证据，不是内核融合或编译提速。回执9项集成检查全部通过，保存源码摘要与环境。没有自然语言训练、GPU性能、生产内存或质量—成本前沿结果。

## 10. 论文 claim 边界与补证优先序

| 当前可用的表述 | 尚未成立的表述 | 决定性补证 |
|---|---|---|
| 历史静态 LM 中几何差异大、任务差异小 | 静态或动态 IsoHC 有普遍性能收益 | 强读写和 gain 对照、faithful 动态框架、配对多 seed |
| 指定 transport 中补空间收缩/保能量可测 | 整网梯度稳定、保存的能量等于有效记忆 | 输入输出/写到读 kernel、任务干预与质量收益联结 |
| 当前优化器对 mixing/gates 施加 WD；独立 WD-only 反事实接近历史均值谱 | 最终谱完全由 WD 决定、transport 无 loss 梯度 | init-only、冻结、no-decay 的匹配轨迹与梯度分解 |
| GNN 架构包明显改善此 Cora 设置 | IsoHC 是改善的唯一原因或 orthogonal 总是更好 | 固定初始化/读写/参数预算、逐 seed 配对与强无 mixing 臂 |
| 附件报告 CPU 等距交换无可检测收益 | 等距交换路线已被否定或已被证明 | 恢复原件，后续在目标任务规模测试预先定义效应 |
| adjoint完整模型可训练、路由可激活、固定地址有精确残差语义 | adjoint具有LM收益、动态稳定性或质量—成本优势 | 强单流、同carrier自由读写、冻结辅助写入和faithful竞争方案的匹配比较 |

下一阶段围绕已经实现的adjoint候选检验实际LM效果、自由读写限制的得失，以及真实GPU成本；完整顺序见 [roadmap.md](roadmap.md)。若冻结辅助流或强单流保留全部收益，优先简化而非继续增加控制器。RDM生命周期工作流不恢复。算法存在、数学性质成立、工程可用和论文贡献是四项不同结论，后续记录必须分别登记。
