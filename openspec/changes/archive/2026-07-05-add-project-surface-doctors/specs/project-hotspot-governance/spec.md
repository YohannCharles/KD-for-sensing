## ADDED Requirements

### Requirement: Hotspot next-touch doctor
项目 MUST 提供 hotspot next-touch doctor 或等价报告，用于对已登记热点 owner 输出下一次触碰时的推荐动作：split、merge、keep-and-test、monitor、accepted-size 或 hard-budget。该判断 MUST 基于 owner 职责、public surface、调用关系、headroom、focused tests 和 inventory rationale，不得仅基于行数。

#### Scenario: 长文件不被机械拆分
- **WHEN** doctor 发现某文件超过行数趋势阈值
- **THEN** doctor MUST 检查该文件是否为已登记 accepted owner、facade、workflow orchestrator 或 split-next 热点
- **AND** 输出 MUST 包含推荐动作和理由，而不是只报告行数

#### Scenario: facade 超预算
- **WHEN** doctor 发现 public facade 吸收 suite-specific helper 或超过 hard-budget 边界
- **THEN** 输出 MUST 标记为高风险
- **AND** 建议动作 MUST 指向把实现移回窄 owner 或删除低价值 facade
