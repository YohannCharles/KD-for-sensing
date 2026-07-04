# research-literature-matrix Specification

## Purpose
定义研究 literature matrix 与可选 BibTeX 关联规则，用于追踪当前多模态/缺失模态 beam prediction 主线的论文、artifact 可用性、repo 关系和 claim 使用边界。
## Requirements
### Requirement: Literature matrix
项目 MUST 维护研究相关 literature matrix，用于记录与当前多模态/缺失模态 beam prediction 主线有关的论文、方法、数据集、模态、缺失协议、指标、开源状态和本仓库对照关系。

#### Scenario: 文献矩阵字段
- **WHEN** 开发者打开 `docs/literature_matrix.md` 或等价文档
- **THEN** 文档 MUST 至少包含 paper id、title/venue/year、method family、modalities、missing-modality protocol、dataset、metrics、official code/data/checkpoint availability、repo relation、claim usage 和 caveat
- **AND** 文档 MUST 区分 official reproduction、local substitute、source audit、background citation 和 not comparable

#### Scenario: 当前主线文献覆盖
- **WHEN** literature matrix 首版落地
- **THEN** 它 MUST 覆盖 DeepSense6G/BeamBench、Vision-Position baseline、AMBER、AMR-Net、RMBP-MM/WCL missing-modality、JEPA/GPS shortcut 或当前主线直接引用的相关工作
- **AND** 缺 DOI、BibTeX 或官方 artifact 时 MUST 标记 pending/unavailable，而不是猜测

### Requirement: BibTeX references
项目 MAY 维护 `paper/references.bib` 或等价 BibTeX 文件。若维护，BibTeX MUST 与 literature matrix 中的 paper id 可关联。

#### Scenario: BibTeX key 可追踪
- **WHEN** literature matrix 行声明 bibtex key
- **THEN** `paper/references.bib` 中 SHOULD 存在同 key 条目
- **AND** 缺失条目 MUST 在文档或 audit 中标记为 pending

#### Scenario: 不下载外部论文
- **WHEN** 更新 literature matrix 或 BibTeX
- **THEN** 系统 MUST 不自动下载 PDF、官方代码、数据或 checkpoint
- **AND** 本地 `paper/*.pdf` 只作为用户已有资料处理
