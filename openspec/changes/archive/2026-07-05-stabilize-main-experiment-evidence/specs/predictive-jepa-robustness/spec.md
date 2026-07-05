## ADDED Requirements

### Requirement: Predictive JEPA real benchmark promotion gate
Predictive JEPA robustness 从 smoke 升级为真实 claim 前 MUST 满足 real benchmark promotion gate。Gate MUST 要求 audited Image ResNet+GPS baseline、JEPA baseline、predictive hybrid checkpoint、clean anchor、`image_missing`、`image_noise`、`gps_noise` stress curves、difficulty digest、seed、split、sample_count、label_space、metric_profile 和 checkpoint provenance。缺少任一关键字段时，claim status MUST 保持 pending、unavailable、mock/smoke 或 not_comparable。

#### Scenario: smoke manifest 不可升级
- **WHEN** benchmark 输入使用 synthetic metrics、mock weights、allow_missing_artifacts 或 partial model set
- **THEN** predictive robustness claim MUST 保持 `mock/smoke`、`pending`、`unavailable` 或 `not_comparable`
- **AND** 系统 MUST 不输出正式 `margin_vs_resnet_dba` claim

#### Scenario: real benchmark 满足 gate
- **WHEN** real manifest 包含 required model groups、strict comparability fields、clean anchor 和默认 stress curves
- **THEN** benchmark MUST 生成 upgradable claim candidate
- **AND** 只有 `margin_vs_resnet_dba` 达到当前阈值且无 clean regression blocker 时，claim doctor 才能提示人工升级
