# LLM residual algorithm research

当前算法候选：**伴随读写残差 `adjoint-hc`**。两条流用同一个单位向量读取和写回，保留identity carry；无需额外混合矩阵、teacher或离散调度。已实现完整Transformer和训练入口，正在验证质量与成本，尚非SOTA结论。

- [算法与可运行实现](docs/research/architecture.md) · [源码](lm/adjoint.py)
- [研究目标](docs/research/README.md) · [理论证明](docs/research/theory.md)
- [训练/工程回执](results/adjoint_hc_20260926/README.md) · [证据台账](docs/research/evidence.md)
- [下一实验](docs/research/roadmap.md) · [先行工作](docs/research/literature.md)
- [文档权威](AGENTS.md) · [登记](docs/research/document_registry.json)

```bash
python3 -m unittest discover -s tests -p 'test_adjoint_contracts.py' -v
python3 experiments/adjoint_hc_probe.py
python3 scripts/check_research_docs.py
```

probe执行小型合成NTP训练和本地CPU性能检查，不下载数据，不作LM优劣判决。正式runner的新增方法与缓存参数见 [experiments/README.md](experiments/README.md)。

RDM已撤销主线资格；旧IsoHC、历史计划与用户源文保留原位，不发布当前执行指令。2027主会目标不变，方法价值由充分训练与强基线决定。
