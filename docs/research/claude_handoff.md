# Claude 交接：历史来源与待恢复资产

来源：用户于 2026-09-25 提供的附件 `已粘贴的文本.txt`，以 “Confirmed gauge equivalence…” 开头。以下忠实保留其科学判断与数字，省略工作过程及权限处理信息。它是交接摘要，不是原始实验记录；本轮的独立评估在 [README](README.md)、[theory](theory.md)、[evidence](evidence.md) 与 [literature](literature.md)。

> 来源摘要，不是当前计划或执行指令。当前研究目标以 [研究主线](README.md) 和 [架构选择](architecture.md) 为准；以下保留附件原判断，不因新方向而改写其内容。

## 资产缺口

附件称结果已保存于分支 `claude/wonderful-gauss-fsuyj9`，提交 `5c047df`，目录 `docs/0924_salvage/`，但推送遭拒。2026-09-25 当前本地分支为 `main`、起点 `d4af4d6`，没有该分支或目录。未连接 Claude 运行环境，未将“附件称已提交”视为本地已有。

待恢复：完整报告、gauge 检查脚本、WD-only 重放脚本、CPU probe 代码、每个 arm 的配置/seed/log/checkpoint/结果、depth-scaled init 对照、原始文献调查定位。恢复后按代码与输出逐项升级证据状态，不覆盖当前独立审计。

## 附件的主要结论（reported-only）

1. 建议停止 IsoHC 作为方法，转向 gauge 理论和 mean↔difference 交换。
2. 静态 IsoHC 可通过旋转读写向量变成 identity-HC；报告实际模型 float32 logits 误差约 `2e-7`、float64 代数误差约 `3e-16`。同时讨论 mHC 的逆变换范数代价。
3. 认为旧实验 `1 + λP⊥w`、`λ=0.01` 的弱读写，使 transport 对 loss 影响小；全参数 AdamW 的 decay 主导 mHC 的谱演化。
4. WD-only 报告：48L 单步谱预测 `0.92174`、实测 `0.92221`；24L 预测 `0.91542`、实测 `0.91563`；48L composite 预测 `4.0e-4`、实测 `4.39e-4`。这些数字的同 run 对齐需要恢复原始记录；拟合衰减曲线不等于证明 loss 梯度为零。
5. 建议用 faithful 动态框架比较 baseline、residual 标量、identity、mHC、IsoHC、SO(n)、exchange、decay、free gain、可学习 exit。

## CPU probe（reported-only）

附件描述约 0.8M 参数、16 层、char-level、1,200 steps、两个 seed，以及 depth-scaled init 对照；未提供本地可审计数据集划分与完整结果表。

- 同配置 seed 差可达约 `0.03` loss；identity、IsoHC、正交、exchange 等无可检测差异。
- unconstrained 相比 identity 好约 `0.05`，free gain 好约 `0.03`；depth-scaled init 后趋势保留。
- 早层主通道增益约 `2–3×`，附件解释为早层/embedding 权重上升。
- 强耦合下移除 complement 可使 loss 增加 `0.2–1.2` nats，原模型相对弱耦合却只好约 `0.01`。

这组结果提出了有价值的控制项：可学习残差增益、初始化、读写幅度，以及“模型依赖某状态”与“该状态有额外任务价值”的区别。两 seed 的结果不足以声明稳健总体收益。

## 不自动继承的决策

附件提出 3–4 周、单张 5090、100–150M / 2–3B tokens、300–500 GPU 小时，以及 `1σ/2σ`、恢复 `70%` 增益、通过概率 `25–35%`。这些是 Claude 的建议或主观估计，不是用户授权、已测工时、统计结论或本项目固定预算。

附件的“保均值使额外 streams 无效”“真作用只有三种”“动态 IsoHC 必须等于 identity 才继续”“loop 是唯一静态非 gauge 场景”均不作为已证命题。新计划逐条处理其成立条件，并允许实验改变具体机制判断。
