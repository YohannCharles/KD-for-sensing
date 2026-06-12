## ADDED Requirements

### Requirement: runtime output 整理 manifest
系统 MUST 提供只读 runtime output 整理 manifest 能力，用于为 `outputs/` 中的历史训练 run、评估 run、registry checkpoint、analysis、cache 和 legacy 目录生成 move/archive/protect/review plan。整理 manifest 生成过程 MUST 不删除、不移动、不压缩、不重写任何本地数据、输出、日志、cache、checkpoint、源码、配置、文档或 OpenSpec artifact。

#### Scenario: 生成整理 dry-run manifest
- **WHEN** 用户对 `outputs/` 运行整理 dry-run
- **THEN** 系统 MUST 写出 machine-readable manifest
- **AND** manifest MUST 记录每个候选的 source path、建议 target path、action、artifact type、size、mtime、匹配原因、风险等级、保护状态和是否需要人工复核
- **AND** 系统 MUST 不移动或删除任何候选路径

#### Scenario: 分类 legacy 输出目录
- **WHEN** 整理扫描发现根级训练 run、`outputs/31/`、根级 `outputs/best_checkpoints/` 或 `outputs/eval_*`
- **THEN** manifest MUST 将它们分类为 legacy root run、legacy numeric scene、legacy registry 或 legacy evaluation
- **AND** manifest MUST 给出 canonical target 或 archive target
- **AND** 无法可靠判断 scope 的候选 MUST 标记为 `review` 或 `protect`

#### Scenario: cache 默认受保护
- **WHEN** 整理扫描发现 `outputs/cache/`
- **THEN** manifest MUST 默认将 cache 分区标记为 protected summary
- **AND** manifest MUST 不建议把 cache 移入训练 run、evaluation、analysis 或 archive

### Requirement: runtime output 整理执行阶段
系统 MAY 提供基于整理 manifest 的执行阶段，但执行阶段 MUST 要求用户显式传入 manifest 和确认参数。执行前 MUST 重新验证每个候选仍未受保护、source 仍在允许根下、source 状态与 manifest 兼容、target 不冲突且不会覆盖已有产物。执行阶段 MUST 写出 execution report。

#### Scenario: 未确认时拒绝整理执行
- **WHEN** 用户调用整理执行阶段但未提供显式确认参数
- **THEN** 系统 MUST 拒绝执行
- **AND** 系统 MUST 提示先检查整理 manifest 并提供确认参数

#### Scenario: 目标冲突时跳过
- **WHEN** manifest 中某个候选的 target path 在执行前已经存在且未声明可安全合并
- **THEN** 系统 MUST 跳过该候选
- **AND** execution report MUST 记录冲突 target 和跳过原因

#### Scenario: 路径变化时跳过
- **WHEN** manifest 中某个候选在执行前 size、mtime、保护状态或 git tracked 状态发生变化
- **THEN** 系统 MUST 跳过该候选
- **AND** execution report MUST 记录状态变化原因
