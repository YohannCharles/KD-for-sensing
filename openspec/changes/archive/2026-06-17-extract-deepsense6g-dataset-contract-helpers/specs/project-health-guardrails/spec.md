## ADDED Requirements

### Requirement: Dataset contract helper 热点治理
项目健康护栏 SHALL 鼓励 DeepSense6G dataset contract helper 拆分，并防止新的契约规则继续堆入 `DeepSense6GDataset` 超长类。热点 inventory 和 maintainer context index MUST 记录 helper 拆分方向和预算。

#### Scenario: DeepSense6GDataset 预算下降或保持有理由
- **WHEN** helper 拆分完成
- **THEN** `docs/maintainer_context_index.yaml` MUST 更新 `DeepSense6GDataset` 和 `__init__` 的热点预算或记录暂缓原因
- **AND** `docs/project_surface_inventory.md` MUST 说明哪些契约 helper 已拆出

#### Scenario: 新契约规则进入 helper
- **WHEN** 后续新增 GPS feature mode、beam target source、column guard 或 cache path rule
- **THEN** 主要实现 MUST 位于 DeepSense6G contract helper 模块
- **AND** 架构或 focused tests MUST 防止这些规则继续扩大 dataset class 主体
