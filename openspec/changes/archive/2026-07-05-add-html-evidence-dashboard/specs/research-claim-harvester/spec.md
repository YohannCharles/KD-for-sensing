## MODIFIED Requirements

### Requirement: Daily research dashboard
系统 MUST 提供 daily research dashboard，用于给研究者快速查看当前实验状态、资源状态和 claim readiness。Dashboard MUST 可输出人类可读文本摘要、机器可读 JSON 和静态 HTML evidence dashboard。文本、JSON 和 HTML 输出 MUST 使用同一 dashboard summary 数据。

#### Scenario: dashboard 聚合运行和 claim 状态
- **WHEN** 用户运行 dashboard 命令
- **THEN** 输出 MUST 至少包含 active OpenSpec change 摘要、running/waiting/failed/stale run 计数、GPU/进程快照、pending/unverified claim 计数、可升级 claim candidate 和缺失的下一步
- **AND** 命令 MUST 保持只读

#### Scenario: dashboard 输出 next action
- **WHEN** 某个 method 缺少 seed、fresh eval、strict field 或 checkpoint provenance
- **THEN** dashboard MUST 输出 next-action hint
- **AND** hint MUST 不自动启动训练、评估、清理或文档修改

#### Scenario: dashboard 写出 HTML evidence report
- **WHEN** 用户指定 HTML dashboard 输出路径
- **THEN** dashboard MUST 写出静态 HTML 文件
- **AND** HTML MUST 与 JSON summary 中的 run state、claim status、warnings、candidate-only caveat 和 next actions 保持一致
- **AND** HTML 输出 MUST 位于 ignored output root 或用户显式路径
