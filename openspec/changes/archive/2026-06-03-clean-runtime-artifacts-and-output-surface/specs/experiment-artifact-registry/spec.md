## ADDED Requirements

### Requirement: checkpoint 保留策略
训练和清理工作流 MUST 区分复现必需 checkpoint、选择指标 checkpoint、恢复训练 checkpoint 和临时 checkpoint。默认清理策略 MUST 保护 `best.pth`、`best_top1.pth`、checkpoint sidecar metadata、归一化 artifacts、metrics、最终配置和 startup summary；`last.pth` 或重复 probe checkpoint MAY 进入候选，但 MUST 记录风险等级和保留理由。

#### Scenario: 默认保护最佳 checkpoint
- **WHEN** 清理 manifest 扫描包含 `checkpoints/best.pth` 或 `checkpoints/best_top1.pth` 的 run
- **THEN** manifest MUST 默认将这些 checkpoint 标记为 protected
- **AND** manifest MUST 保留对应 sidecar metadata 的保护关系

#### Scenario: last checkpoint 可作为候选
- **WHEN** run 已完成且存在 `checkpoints/last.pth`，同时存在受保护的最佳 checkpoint 和 metrics
- **THEN** manifest MAY 将 `last.pth` 列为可删除候选
- **AND** manifest MUST 记录该候选不是默认复现 checkpoint

### Requirement: checkpoint retention metadata
运行产物摘要 MUST 能表达 checkpoint retention 决策所需 metadata。系统 MUST 记录 checkpoint 来源、选择指标、selected epoch、run 状态、是否有 sidecar、是否有归一化 artifact 引用和是否属于 registry 默认候选。

#### Scenario: retention 摘要包含选择信息
- **WHEN** run index 或清理 manifest 汇总 checkpoint
- **THEN** summary MUST 包含 checkpoint 来源和 selection metadata（如果可用）
- **AND** 缺失 metadata 时 MUST 记录缺失状态而不是推断为可删除
