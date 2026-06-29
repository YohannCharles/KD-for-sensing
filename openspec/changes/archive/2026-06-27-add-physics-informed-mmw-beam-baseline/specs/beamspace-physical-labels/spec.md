## ADDED Requirements

### Requirement: Beamspace labels as physics loss supervision
系统 SHALL 允许 physics-informed baseline 将 `beamspace_power_label`、beam power vector 或 path-based beamspace fallback 作为 beam distribution/array consistency 的监督来源。该监督 MUST 保持 existing class calibration、availability mask、cache metadata 和 target-domain leakage boundary。

#### Scenario: 使用 beamspace power 监督 physics logits
- **WHEN** batch 包含有效 `beamspace_power_label` 且配置启用 beam distribution loss
- **THEN** loss MUST 将 physics logits 或 hybrid logits 与该分布对齐
- **AND** diagnostics MUST 记录有效 horizon 数、source 类型和 class order calibration fingerprint

#### Scenario: beamspace label 不可用时跳过分布 loss
- **WHEN** `beamspace_power_available=false`
- **THEN** beam distribution loss MUST 返回零贡献
- **AND** diagnostics MUST 记录 unavailable reason
- **AND** hard beam CE MUST 继续按 `target_beam` 计算

#### Scenario: target-domain boundary 保持不变
- **WHEN** target adaptation batch 包含 `beamspace_power_label`
- **THEN** 默认 adaptation loss MUST 不使用 target-side beamspace label 反传
- **AND** physics-informed metadata MUST 记录该字段仅用于评估诊断或明确的 source supervision
