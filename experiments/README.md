# 实验入口与当前能力边界

更新：2026-09-29，R5已按用户要求收尾，实例已关机；下列入口保留为研究工具示例，不恢复本轮旧计划。决定和实测结论见 [收尾报告](../docs/research/reports/R5_CLOSEOUT_20260929.md)。当前研究合同在 [研究主线](../docs/research/README.md)；R4原入口与回执保留作参考。

## R5诊断入口

[phase实现](../lm/phase_adjoint.py)与[Block AttnRes机制对照](../lm/block_attnres.py)使用相同骨干。默认core7为baseline、gain、phase-adjoint、terminal-adjoint、boundary-skip、phase-adjoint-post-frozen、block-attnres。terminal只在全部body之后交汇，已通过独立history梯度与切换时机合同。shear、全frozen负控制和R4另有明确method ID。

```bash
python3 experiments/verify_phase_initialization.py
python3 -m unittest discover -s tests -p 'test_phase_adjoint.py' -v
python3 -m unittest discover -s tests -p 'test_residual_runner.py' -v
python3 experiments/prepare_residual_data.py \
  --output-dir data/residual_fwe300m --train-tokens 300000000 --val-tokens 2000000
python3 experiments/residual_lm_diagnostic.py \
  --methods baseline gain phase-adjoint terminal-adjoint boundary-skip phase-adjoint-post-frozen block-attnres \
  --train-cache data/residual_fwe300m/train.pt --val-cache data/residual_fwe300m/val.pt \
  --vocab-size 50257 --context-length 512 --layers 24 --width 256 --heads 4 \
  --micro-batch 32 --grad-accum 1 --updates 16384 --warmup-updates 256 \
  --eval-every 2048 --checkpoint-every 2048 --val-blocks 3906 --model-seed 419 --data-seed 20260929 \
  --device cuda --compile --output-dir outputs/r5_development
```

这是单LR合同示例；实际双LR选择/续跑协议见 [roadmap](../docs/research/roadmap.md)。9月29日GPU profile支持micro32×accum1（有效batch32），比初始micro8×accum4更适合该卡。编译训练固定casts emulation及backward autocast off，持出评价统一raw eager BF16，数值策略进入resume身份。16384 updates对应268,435,456训练tokens/臂，validation为1,999,872个不重叠target。这是开发诊断，不是成熟LM证据。

训练cache必须是明确的1D整数.pt，旁边有complete manifest并匹配vocab、dtype、实际tokens及文件SHA。准备程序按文档内容SHA分流、固定HF revision、GPT-2 ordinary编码加EOS；可用`--revision`固定官方已核验SHA。`HF_ENDPOINT`只改变传输端点，dataset和revision不改变；临时镜像路径不作为新数据源。

`--dry-run`不训练；`--profile-steps`是独立CUDA完整AdamW测量，不复用其更新进入科学训练。`--updates`冻结全局LR schedule，`--stop-after`在相同schedule上中断；恢复用单method、原配置与`--resume .../final.pt`。严格拒绝配置、cache、源码或Torch版本变更，保存原始model键、optimizer/scaler/RNG和实际数据cursor。恢复累计成本还须保留同tag的`final.pt.timing.json`；每个验证点保存跨恢复的累计时钟。sidecar缺失/错配仍可恢复模型，但标为timing incomplete，不能用于完整成本曲线。

全臂统一残差输出初始化为1/sqrt(2L)，矩阵WD与norm/router/gate no-WD分开。旧runner的epoch预算、重叠评价和恢复缺口不在此路径继承。Block AttnRes使用参考PyTorch实现，保持原算法的源拓扑；其吞吐不能代表生产融合kernel，也不将其source mean初始化说成逐位baseline一致。

[顺序诊断驱动](run_residual_study.py)支持固定七臂双LR前缀及选中轨迹的resume；先生成`--dry-run`计划，核对后用同参数`--execute-reviewed-plan`。传入实际profile速率文件、deadline、开销/余量和按method复用的compile cache；不能直接套用别的GPU速率。开始任一primary后冻结LR选择；身份错配拒绝复用，子进程退出未确认则中止调度。该驱动管理实验，不是模型内部controller。本轮 [study plan](../results/phase_adjoint_20260929/study_plan.json)已停止，不作为下次默认任务。[分析工具](analyze_residual_study.py)验证源码、配置、数据、评价及预算，缺失/中断臂不得进入完整预算排名。

GPU启动预检顺序为：七臂同shape完整训练profile；七臂完整验证、保存；baseline/phase恢复。验证3906 blocks在micro32下包含122个B32及一个B2尾批次，使用共同raw eager evaluator。可在独立预检目录用`--updates 16384 --stop-after 1 --val-blocks 3906`检查。默认compiled no-grad评价已由CUDA数值核验排除出科学NLL路径，旧preflight权重不用于正式轨迹；训练身份冻结后不能无声切换编译方式。

LR开发段保留完整schedule：`--updates 16384 --stop-after 2048`，按相同身份恢复选定臂。3e-4/6e-4若保持共同终末LR比例，分别显式使用`--min-lr 3e-5`/`--min-lr 6e-5`。先调baseline，是否给core6等额两LR开发由实测吞吐与明确费用上限决定；共用baseline选定配方只能支持开发比较。完整D1每臂实际17次eval和17次save（含末尾重复），这些成本一起计入，不能仅报steady train时间。

## R4参考算法：adjoint-hc

[完整模型](../lm/adjoint.py)复用baseline主干，增加一条辅助流，以同一个动态单位地址读取和写回。初始化与共享权重baseline完全相同；signed carrier提供初始路由学习信号。

```bash
python3 -m unittest discover -s tests -p 'test_adjoint_contracts.py' -v
python3 experiments/adjoint_hc_probe.py --compile-check
```

已完成16项合同测试（含正式runner的4步真实训练）及四臂各80步合成NTP训练。[回执](../results/adjoint_hc_20260926/README.md)保存完整配置、曲线和两个CPU成本测量。合成任务各臂loss几乎重合，不构成LM质量优势。`aot_eager`全图前反向成功不代表已有融合内核。

`lm_5090_next_runs.py`已注册以下方法，全部明确使用两流：

| 方法ID | 区别 |
| --- | --- |
| `adjoint-hc` | 动态单位地址，signed carrier，同方向写回 |
| `adjoint-hc-static` | 仅学习各子层的标量地址 |
| `adjoint-hc-frozen-aux` | 保留同样的辅助入口和读取，禁止后续辅助写入 |
| `adjoint-hc-zero-carrier` | 辅助入口为零，验证对齐初始化的路由死区 |
| `adjoint-hc-copy-carrier` | 辅助入口为x₀，比较符号变换的作用 |

R4入口的正式数据通过既有`--train_cache_path`、`--val_cache_path`参数加载；方法用`--methods`，配置用`--preset`选择。不要把旧四流配置或head-mix preset直接套到新方法。R4回执当时没有下载语料或执行GPU训练；9月29日R5的远端300M/2M cache已完成，使用本页前述独立runner。

旧训练器的全参数AdamW分组保持原样。正式研究需显式记录并公平处理各臂的优化器分组与调参预算，不能将可运行入口当作已优化的大规模训练配方。同carrier的两流自由读写、强单流gain和faithful动态竞争方法尚需匹配实现。

## 当前可直接运行的离线核验

```bash
python3 experiments/verify_gauge_contracts.py
python3 -m unittest discover -s tests -p 'test_transport_composition.py'
```

[回执与解释](../results/theory_audit_20260925/README.md)：实际模型 gauge logits、边界与奇异反例、深度核不变量、保均值 rank-2 构造、WD-only 对照。数值 seed 是代数例子，不是独立训练重复。

## 当前工程状态

完整adjoint/phase/terminal模型已有forward/backward、低精度、因果性与batch隔离检查，以及CPU合成优化和恢复合同。R5已经进行真实语料GPU优化步profile、默认策略完整验证/保存及CUDA数值核验；正式质量轨迹另行记录。生产kernel、成熟LM质量、decode和分布式收益尚未验证。

RDM已撤销主线资格。[历史slot probe](../results/depth_memory_contracts_20260925/README.md)可通过 `python3 experiments/verify_depth_memory_contracts.py` 重现；它没有学习控制器、完整模型或GPU工程证据，不是当前默认下一实验。旧数据与代码继续保留。

## 历史 LM 资产

| 入口 | 能做什么 | 边界 |
| --- | --- | --- |
| `run_0525_mechanism_gpu_pipeline.sh` | 旧 FE48 GPU pipeline、checkpoint/posthoc | 历史配方，不是下一阶段默认命令 |
| `hc_causal_controls.py` | 2026-07 P0/P1几何、因果与可达性控制套件 | 历史控制工具；背景材料见已标记历史状态的0714/0715文档，不是当前默认实验 |
| `lm_5090_next_runs.py` 的旧方法ID | baseline、identity-HC、mHC、IsoHC、unconstrained | 旧HC的H与读写参数静态；新增adjoint动态方法单列于上方，不能将旧proxy标为faithful动态mHC |
| `analyze_lm_mechanisms.py` | 原 checkpoint 的谱、梯度、删除及替换分析 | 原始结果身份需对齐；直接替换 H 不是 gauge 变换 |
| `prepare_lm_data.py` | tokenizer/token-cache 准备 | 数据准备/下载与训练执行另按实际任务范围安排 |
| `stage1_*`, `stage2_*`, `gnn_*` | 算子正确性、precision、toy/GNN 探索 | 不替代语言模型的任务收益 |

模型本身已有 `OrthogonalMixing`，但旧主 runner 没有提供完整的动态 orthogonal/exchange 训练合同。9月26日R4新增的是明确列出的五个adjoint方法；9月29日R5已实现本页前述phase及强对照、Block AttnRes机制比较器。后者不是生产融合kernel复现，其余未实现方法仍不能写成已可运行。

## 诊断版本

`lm/transport_analysis.py` 的 schema 2 使用完整矩阵乘积后投影：`Uᵀ(H_k…H_1)U`。`projected_step_product_sv_*` 单列历史“每步投影后相乘”的量。两者在精确保补空间条件下相同；允许 mean↔difference 交换时不同。旧非保补空间结果若用于新论证需要重算，历史文件保留原样。

## 数据与服务器约定

旧环境惯例：源码 `/root/isoHC`，大型数据、cache、结果 `/root/autodl-tmp/isoHC`；SSH 端口随实例变化，必须用当前明确提供的 endpoint。根目录 [agent.md](../agent.md) 保留存储约定。这些历史信息不表示当前服务器在线或有训练授权。

已删除的 legacy FE/TinyShakespeare 启动脚本不恢复为主入口；其历史记录仍在 `docs/0605_alldoc/`。GNN 多 seed 摘要的一部分只在本地 ignored `results/`，关键数值和证据等级已登记到当前证据页。
