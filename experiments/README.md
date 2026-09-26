# 实验入口与当前能力边界

更新：2026-09-26，R4。当前研究合同在 [研究主线](../docs/research/README.md) 和 [下一阶段计划](../docs/research/roadmap.md)。当前已实现伴随读写残差；以下区分新算法集成训练、离线核验与历史LM训练。

## 当前算法：adjoint-hc

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

正式数据仍通过既有`--train_cache_path`、`--val_cache_path`参数加载；方法用`--methods`，配置用`--preset`选择。不要把旧四流配置或head-mix preset直接套到新方法。当前本地数据目录为空；本轮没有下载语料或执行GPU训练。

旧训练器的全参数AdamW分组保持原样。正式研究需显式记录并公平处理各臂的优化器分组与调参预算，不能将可运行入口当作已优化的大规模训练配方。同carrier的两流自由读写、强单流gain和faithful动态竞争方法尚需匹配实现。

## 当前可直接运行的离线核验

```bash
python3 experiments/verify_gauge_contracts.py
python3 -m unittest discover -s tests -p 'test_transport_composition.py'
```

[回执与解释](../results/theory_audit_20260925/README.md)：实际模型 gauge logits、边界与奇异反例、深度核不变量、保均值 rank-2 构造、WD-only 对照。数值 seed 是代数例子，不是独立训练重复。

## 当前工程状态

完整adjoint模型已有forward/backward、BF16状态、因果性与batch隔离检查，以及CPU合成训练和成本记录。GPU融合、真实语料训练、生产显存与端到端速度尚未验证；按 [架构选择](../docs/research/architecture.md) 与 [实验路线](../docs/research/roadmap.md)推进。

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

模型本身已有 `OrthogonalMixing`，但旧主 runner 没有提供完整的动态 orthogonal/exchange 训练合同。本轮新增的是明确列出的五个adjoint方法；Block AttnRes及其他计划对照不能写成已可运行。

## 诊断版本

`lm/transport_analysis.py` 的 schema 2 使用完整矩阵乘积后投影：`Uᵀ(H_k…H_1)U`。`projected_step_product_sv_*` 单列历史“每步投影后相乘”的量。两者在精确保补空间条件下相同；允许 mean↔difference 交换时不同。旧非保补空间结果若用于新论证需要重算，历史文件保留原样。

## 数据与服务器约定

旧环境惯例：源码 `/root/isoHC`，大型数据、cache、结果 `/root/autodl-tmp/isoHC`；SSH 端口随实例变化，必须用当前明确提供的 endpoint。根目录 [agent.md](../agent.md) 保留存储约定。这些历史信息不表示当前服务器在线或有训练授权。

已删除的 legacy FE/TinyShakespeare 启动脚本不恢复为主入口；其历史记录仍在 `docs/0605_alldoc/`。GNN 多 seed 摘要的一部分只在本地 ignored `results/`，关键数值和证据等级已登记到当前证据页。
