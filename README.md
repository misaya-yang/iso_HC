# LLM residual algorithm research

**R5诊断已关闭（2026-09-29）**。同seed短段比较中，gain在两档LR均优于phase，成本接近baseline；268M-token主预算未完成。`phase-adjoint`与`terminal-adjoint`保留为研究资产，当前没有批准的新候选，不自动恢复或扩规模。R4 `adjoint-hc` 保留为参考，RDM保持归档。

- [R5关闭报告](docs/research/reports/R5_CLOSEOUT_20260929.md)
- [研究目标与当前决定](docs/research/README.md)
- [算法与强skip对照](docs/research/architecture.md) · [证明和反例](docs/research/theory.md)
- [有限GPU诊断与决定](docs/research/roadmap.md) · [最新先行性](docs/research/literature.md)
- [证据与执行状态](docs/research/evidence.md) · [文档权威](AGENTS.md)
- [保留实现](lm/phase_adjoint.py) · [Block AttnRes对照](lm/block_attnres.py)
- [数据准备](experiments/prepare_residual_data.py) · [可恢复训练入口](experiments/residual_lm_diagnostic.py)

以下命令仅核验保留资产：

```bash
python3 experiments/verify_phase_initialization.py
python3 -m unittest discover -s tests -p 'test_phase_adjoint.py' -v
python3 scripts/check_research_docs.py
```

NeurIPS2027/ICML2027仍是长期研究目标，当前没有可承诺的投稿候选或SOTA结果。数学合同、实现可训、自然语言收益和论文贡献分别登记。历史数据不改写，旧计划不自动触发执行；后续训练需新的明确授权，并使用有完整manifest的本地cache。
