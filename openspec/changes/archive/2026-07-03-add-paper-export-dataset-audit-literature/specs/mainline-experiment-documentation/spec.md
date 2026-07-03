## ADDED Requirements

### Requirement: Paper export and literature documentation
主线实验文档 MUST 索引 paper artifact export、dataset audit 和 literature matrix，并说明它们的 claim 状态边界。

#### Scenario: 文档索引 paper export
- **WHEN** README 或 `docs/experiment_matrix.md` 新增 paper export 说明
- **THEN** 文档 MUST 说明 export 消费已审阅 claim、ledger 或 summary
- **AND** 文档 MUST 说明 pending/mock/historical rows 默认不进入 main table

#### Scenario: 文档索引 dataset audit
- **WHEN** README_REPRODUCE 或主线文档给出数据审计入口
- **THEN** 文档 MUST 使用当前存在的 audit entrypoint
- **AND** 文档 MUST 说明 audit 只读、不移动数据、不代表 official reproduction 已完成

#### Scenario: inventory 记录 literature matrix
- **WHEN** 新增 `docs/literature_matrix.md` 或 `paper/references.bib`
- **THEN** `docs/project_surface_inventory.md` MUST 记录其文档生命周期和职责
- **AND** 文档 MUST 不把本地 PDF 或外部论文下载物纳入源码产物要求
