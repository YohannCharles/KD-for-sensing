## ADDED Requirements

### Requirement: MMW profile 与结构候选 provenance 必须匹配
MMW training、checkpoint metadata、evaluation worker 和 summary MUST 记录 training profile id、profile fingerprint、T2 design candidate id 与 resolved recipe fingerprint。比较同一方法的多 seed 或同一 summary 行时，任一这些身份不一致 MUST fail closed。

#### Scenario: 拒绝混合 H0/H4 或不同候选
- **WHEN** summary 接收 profile 或 candidate fingerprint 不一致的输入
- **THEN** summary MUST 标记该比较不可用或抛出校验错误
- **AND** 不得通过缺失字段、默认值或人工补值继续聚合
