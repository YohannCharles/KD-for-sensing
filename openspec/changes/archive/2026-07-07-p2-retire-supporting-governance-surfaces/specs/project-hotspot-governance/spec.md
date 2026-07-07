## ADDED Requirements

### Requirement: P2 支持面退役必须有消费者证据
对 LOSO helper、surface doctor default dump、whole-model token transformer、architecture summary/export 等 P2 支持性表面，implementation MUST 基于 current consumer 审计决定删除、瘦身或保留。不得仅凭静态入站少或文件行数大删除 current contract。

#### Scenario: 删除项有无消费者证据
- **WHEN** P2 implementation 删除一个支持性 helper、CLI mode 或 export format
- **THEN** tasks 或 inventory MUST 记录 current docs/spec/tests/config/script consumer 审计结果
- **AND** 删除后 architecture boundary 或 focused tests MUST 防止同职责表面无理由回流

#### Scenario: 保留项有 retained-with-reason
- **WHEN** P2 候选仍有 current consumer 或迁移收益不明确
- **THEN** implementation MUST 保留它
- **AND** inventory MUST 记录保留理由、owner、当前消费者和未来删除触发条件

### Requirement: P2 瘦身不得制造新治理工具
P2 implementation MUST 优先收缩默认输出、删除无消费者格式或迁移 owner，而不是新增更多治理 CLI、wrapper 或 report generator。

#### Scenario: surface doctor 缩短默认输出
- **WHEN** project health guardrail 输出太长
- **THEN** implementation SHOULD 修改默认输出和显式 dump flag
- **AND** MUST 不新增另一个只过滤 surface doctor 输出的 wrapper script
