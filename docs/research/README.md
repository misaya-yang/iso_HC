# 面向2027的残差算法研究

更新：2026-09-29，**R5诊断已关闭**。`phase-adjoint`与`terminal-adjoint`保留为研究资产，不再是当前推进候选；没有批准的新候选，也不自动恢复训练或扩规模。关闭原因、未完成部分及运行身份见 [R5关闭报告](reports/R5_CLOSEOUT_20260929.md)。长期目标仍是强对照之外、有效且简单的残差接口与质量—成本收益。

同seed419、已完成的33.554M-token短段诊断中，LR为3e-4／6e-4时，baseline持出NLL为5.412556／5.124124，gain为5.387013／5.065748，phase为5.391525／5.086086。gain在两档LR均胜phase且成本接近baseline，phase本轮实测约慢18%；这些结果未支持继续优先推进phase。terminal的LR3结果为5.391318，LR6中断，没有可排序的质量点。268M-token主预算没有完成，不能把本次关闭推广为整个方法族失败。

## 保留构造与已验证合同

R4的 `adjoint-hc` 已实现，但全对齐单位Gram核的初始历史导数为0；signed carrier打开的梯度主要来自输入。R5构造从零辅助流开始，在中间block只做一次 \((h,m)\leftarrow(h+m,m-h)\)，其余每个attention/MLP仍只调用一次原branch。零router下初始函数保持baseline，早层写入的一阶损失梯度为 \(\langle g_{boundary},\delta_j\rangle\)。公式、坐标缩放与边界见 [architecture](architecture.md) 和 [theory §11](theory.md)。

独立代数核验已支持初始化和history梯度机制；完整模型、runner及部署状态由 [evidence](evidence.md) 逐项记录。没有因一条公式成立就建立自然语言优势、整网稳定或首次性。

## 本轮对照与保留问题

一个gated boundary skip有相同的一阶history梯度。`terminal-adjoint`仅在全部body之后交汇，就让全部body writer获得真实历史的一阶梯度，因而内部交汇本身也需要证据。本轮设定的对照包括调好的单流、非零一阶gain、内部phase、末端terminal、同梯度boundary skip、只冻结post辅助写入及Block AttnRes；自由shear和R4也保留为机制控制。设定的比较不等于全部预算已经完成。

保留的科学问题是：**在让真实历史容易学习之后，更早使用历史、可更新视图与读写对齐是否仍提供值得采用的额外作用。** 它仍属于HC的受限参数化，不宣称一个此前不存在的函数类。§11.8的参考地址维度刻画和§11.9的末端反例共同约束设计，均不代替质量结果，也不构成重启授权。

## 论文与有限算力的关系

本轮使用约32M深模型、普通自然语言NTP和实测GPU成本诊断，没有进入100–150M及独立尺度确认。NeurIPS2027/ICML2027仍是长期目标，当前没有可承诺的投稿候选或SOTA结果。后续选题与训练须重新明确授权；充分训练、强对照、跨条件复核和真实质量—成本仍是论文贡献的要求。

runner与FineWeb-Edu本地cache保留，预算、数据顺序、非重叠评价与精确resume仍是显式合同。9月29日执行已结束，机器于23:51 UTC确认关闭；本轮启动授权不自动延续到恢复或扩规模。实际运行身份见关闭报告，原实验路线及判断依据见 [roadmap](roadmap.md)。

## 保留资产与唯一owner

R4实现与四臂tiny合成训练保持原样作参考。RDM不恢复；旧IsoHC、gauge、谱和压缩保留为历史证据或控制。共享query bank在本轮审查中对应factorized attention和SANA/MHAR邻域，不并行扩成主线；最新先行性边界见 [literature](literature.md)。

[架构](architecture.md)负责算法，[理论](theory.md)负责证明，[证据](evidence.md)/[登记](evidence_registry.json)负责完成事实，[计划](roadmap.md)负责实验决定，[文献](literature.md)负责先行性。更新owner而不再添加平行“最终方案”。遵循 [AGENTS.md](../../AGENTS.md)，每次文档变更运行 `python3 scripts/check_research_docs.py`。
