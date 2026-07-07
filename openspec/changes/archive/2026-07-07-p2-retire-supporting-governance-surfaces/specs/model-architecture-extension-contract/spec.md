## ADDED Requirements

### Requirement: Whole-model token transformer 删除必须等待 config 迁移
whole-model token transformer fusion MAY 退役，但 implementation MUST 先迁移或移除所有 current registry、config、docs、tests 和 recipe consumer。若任何 canonical config 或 documented workflow 仍引用旧 model key，implementation MUST 保留 whole-model implementation 并记录 retained-with-reason。

#### Scenario: configs 全部迁移后删除
- **WHEN** modular sequence/token core owner 已覆盖 whole-model token transformer 的 current behavior
- **AND** registry、canonical configs、tests 和 docs 都不再引用旧 model key
- **THEN** implementation MAY 删除 whole-model token transformer implementation
- **AND** architecture/config tests MUST 覆盖删除后的 registry 边界

#### Scenario: 仍有 current config 时保留
- **WHEN** 任一 canonical config、paper-facing recipe 或 focused test 仍引用 whole-model token transformer
- **THEN** implementation MUST 不删除该实现
- **AND** inventory MUST 记录保留理由和迁移触发条件

### Requirement: 模型架构 summary/export 只保留当前消费者需要的格式
模型架构 summary/export 支持面 SHOULD 收缩到启动摘要、必要 guardrail 和 current docs/tests 消费的最小格式。删除某个 export format 前 MUST 审计消费者。

#### Scenario: 无消费者 export format 可删除
- **WHEN** 某个 architecture summary/export format 没有 current docs、tests、CLI、CI 或 paper-facing consumer
- **THEN** implementation MAY 删除该 format 或 helper
- **AND** docs 和 help text MUST 不再推荐该 format

#### Scenario: 当前消费者格式保留
- **WHEN** startup summary、architecture guardrail 或 docs 仍消费某个 summary format
- **THEN** implementation MUST 保留该 format 或提供同 owner 下的等价输出
- **AND** 不得新增跨领域 export wrapper 作为替代
