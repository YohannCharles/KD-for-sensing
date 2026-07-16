## ADDED Requirements

### Requirement: MMW 汇总必须验证完整 evidence identity
MMW matrix 与 paired summary MUST 验证 method、seed、profile/candidate/config fingerprint、checkpoint role、sample checksum、mask identity、metric profile 和实际 coverage。缺失、重复或不一致的行 MUST 被拒绝，不能以字典覆盖或默认值聚合。

#### Scenario: 输入包含 partial 或重复 row
- **WHEN** summary 接收 partial output 或相同 identity 的重复 row
- **THEN** summary MUST 报出不可用原因并拒绝正式聚合
- **AND** 不得生成 supported comparison status

### Requirement: 同名 DBA 必须保持统一定义
所有写作 `adba` 的 current MMW evidence MUST 使用 progressive top-3 DBA；任何 top-1 proximity DBA MUST 使用不同的显式字段和 metric profile。

#### Scenario: 汇总两种 DBA 定义
- **WHEN** summary 接收不同 metric profile 的 row
- **THEN** summary MUST 拒绝将它们放入同一 comparison
- **AND** 错误信息 MUST 指出冲突的 metric profile
