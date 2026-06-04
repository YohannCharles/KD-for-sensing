## ADDED Requirements

### Requirement: Beamspace physical labels match calibrated class order
当 MMW beam label calibration 启用时，系统 MUST 将作为监督或评估诊断消费的 `beamspace_power_label` 重排到 calibrated class order，并 MUST 在 cache metadata 中记录 mapping fingerprint。

#### Scenario: beamspace_power_label 重排
- **WHEN** dataset 从 raw beam power vector 构造 `beamspace_power_label` 且 calibration 已启用
- **THEN** 输出 `beamspace_power_label` MUST 满足 `label_calibrated[mapping(raw)] = label_raw[raw]`
- **AND** 每个有效 horizon 的分布和 MUST 在数值容差内等于 1

#### Scenario: physical label cache mapping mismatch
- **WHEN** 已存在 physical label cache 但其 metadata 的 mapping fingerprint 与当前 calibration 不一致
- **THEN** dataset MUST rebuild the cache or reject reuse with a clear error
- **AND** system MUST NOT silently consume raw-order physical labels as calibrated-order labels

#### Scenario: target-domain leakage boundary 保持不变
- **WHEN** target adaptation batch 包含 calibrated `beamspace_power_label`
- **THEN** default adaptation loss MUST NOT use target-side physical labels for backpropagation
- **AND** leakage diagnostics MUST preserve the existing target-domain physical oracle boundary
