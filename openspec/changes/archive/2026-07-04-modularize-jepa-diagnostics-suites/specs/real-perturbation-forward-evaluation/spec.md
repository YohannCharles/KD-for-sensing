## ADDED Requirements

### Requirement: Real-forward diagnostics 必须从 runner 主流程分离
Real-forward perturbation evaluation MUST 在将 condition execution、diagnostics collection 和 cache iteration 移入窄 helper 模块或类时保持 shard/cache/provenance 行为兼容。

#### Scenario: real-forward cache 兼容
- **WHEN** benchmark manifest uses real-forward evaluation
- **THEN** cache keys, sample ids, shard metadata, warning records and unavailable-condition diagnostics MUST remain compatible with existing runner manifests
- **AND** dry-run validation MUST 不要求读取真实 dataset samples
