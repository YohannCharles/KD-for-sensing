## ADDED Requirements

### Requirement: 运行索引提供清理摘要
实验运行索引 MUST 保持只读，并为本地产物清理提供结构化摘要。每个 run summary SHOULD 包含 run 目录大小、checkpoint 文件数量、checkpoint 总大小、日志关联、最近更新时间、状态、可复现关键 artifact 是否存在和清理候选理由。

#### Scenario: run index 输出大小与 checkpoint 摘要
- **WHEN** 用户构建 run index 并扫描 `outputs/`
- **THEN** 每个 run summary MUST 记录 run 目录总大小
- **AND** 如果存在 checkpoint，summary MUST 记录 checkpoint 数量、总大小和主要 checkpoint 路径

#### Scenario: run index 保持只读
- **WHEN** 清理 manifest 生成流程复用 run index
- **THEN** run index MUST 不删除、不移动、不压缩、不重写任何输出、日志、checkpoint 或 cache
- **AND** run index MUST 仅返回结构化摘要

### Requirement: 活跃运行保护信号
实验运行索引 MUST 为清理流程提供活跃运行保护信号。状态为 `running`、`waiting` 或最近仍在更新且无法判定完成的 run MUST 被清理 manifest 标记为 protected 或 high-risk。

#### Scenario: running run 受保护
- **WHEN** run index 通过状态文件、进程或日志判断某个 run 仍在运行
- **THEN** 清理 manifest MUST 不将该 run 列为默认可删除候选
- **AND** manifest MUST 记录活跃运行保护原因

#### Scenario: stale run 可进入人工确认候选
- **WHEN** run index 判断某个 run 超过 stale 阈值且缺少完成指标
- **THEN** 清理 manifest MAY 将该 run 列为人工确认候选
- **AND** manifest MUST 记录 stale 阈值和缺失 artifact 摘要
