## ADDED Requirements

### Requirement: 真实 checkpoint manifest 严格可比
JEPA GPS shortcut benchmark 的真实 checkpoint manifest MUST 为每个模型组声明 checkpoint path、config path、split、sample_count、label_space、metric_profile、difficulty digest、normalization provenance 和 seed。缺失或不一致时，benchmark MUST 将对应 comparison 标记为 unavailable、pending 或 not_comparable，而不是填充 mock 数值。

#### Scenario: checkpoint path 缺失
- **WHEN** real benchmark manifest 中某模型 checkpoint path 不存在
- **THEN** benchmark MUST 将该模型或 comparison 标记为 unavailable
- **AND** 其它可用模型 MUST 继续生成可用的诊断输出，但 claim gate MUST 保持 pending 或 not_comparable

#### Scenario: split 或 metric profile 不一致
- **WHEN** 两个模型组的 split、label_space 或 metric_profile 不一致
- **THEN** shortcut benchmark MUST 标记该 comparison 为 not_comparable
- **AND** paper/claim 输出 MUST 保留 caveat
