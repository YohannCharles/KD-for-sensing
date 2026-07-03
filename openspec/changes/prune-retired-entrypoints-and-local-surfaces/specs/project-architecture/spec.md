## ADDED Requirements

### Requirement: Surface pruning preserves current user behavior
项目 MAY 大规模删除旧入口、本地脚本、隐藏 CLI、重复 tombstone 和可生成配置，但 MUST 保持 current package CLI、current canonical config、dataset split、beam label/label-space、metric schema、checkpoint schema、run metadata 和默认本地产物分区兼容。

#### Scenario: Current public behavior unchanged
- **WHEN** 本 change 删除或合并 internal surface
- **THEN** README、pyproject console scripts、current specs 和 inventory 登记的 current workflow MUST 继续可用
- **AND** 删除 MUST 不要求用户改用未记录的新命令

#### Scenario: Internal breaking import allowed
- **WHEN** 一个 import path 未登记为 public surface
- **THEN** 它 MAY 被删除或移动
- **AND** 项目 MUST 不新增旧路径 compatibility wrapper
