## ADDED Requirements

### Requirement: 退役研究线 checkpoint 可进入清理候选
checkpoint 保留策略 MUST 区分当前主线复现必需 artifact 和已退役研究线产物。退役 Hist/P3/V8/V9 run 中的 checkpoint MAY 进入清理候选，但 manifest MUST 记录 checkpoint 类型、sidecar 状态、run 状态和是否有保留理由。

#### Scenario: 退役 Hist checkpoint 候选记录完整
- **WHEN** 清理 manifest 扫描到退役 Hist run 中的 checkpoint
- **THEN** manifest MUST 记录 checkpoint 文件名、大小、是否为 `best.pth` 或 `last.pth`、是否有 sidecar metadata 和源 run 目录
- **AND** manifest MUST 不因 checkpoint 位于退役 run 中就绕过保护状态检查

#### Scenario: 当前主线 best checkpoint 默认保护
- **WHEN** 清理 manifest 扫描到当前主线 scene-level `best_checkpoints` 或当前主线 run 的 `best.pth`、`best_top1.pth`
- **THEN** manifest MUST 默认将其标记为 protected
- **AND** 删除阶段 MUST 跳过这些 protected checkpoint
