## ADDED Requirements

### Requirement: 验证 hotspot 行动元数据
项目健康护栏 SHALL 验证维护上下文索引中的 hotspot action metadata。检查 MUST 覆盖 priority/status 合法性、split target 列表类型、validation command 环境约束和 inventory marker 对齐。

#### Scenario: hotspot metadata 缺字段
- **WHEN** hotspot budget entry 缺少 priority、status、split targets、rationale 或 validation commands
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向缺失字段

#### Scenario: hotspot validation command 未使用环境
- **WHEN** hotspot metadata 中的 Python/pytest validation command 未使用 `conda run -n kd_mm_beam`
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 指向对应 hotspot entry
