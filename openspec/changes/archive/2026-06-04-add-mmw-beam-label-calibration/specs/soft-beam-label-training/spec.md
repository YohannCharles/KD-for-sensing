## ADDED Requirements

### Requirement: Soft beam labels follow calibrated topology
当 MMW dataset 启用 beam label calibration 且 soft beam label 启用时，系统 MUST 在 calibrated label space 中生成或重排 `target_beam_distribution`，并 MUST 保持该分布与 hard `target_beam` 的 horizon 和 class order 一致。

#### Scenario: Gaussian soft label 使用 calibrated label
- **WHEN** target-domain soft label 基于 hard label 和 circular Gaussian 生成，且 MMW calibration 已启用
- **THEN** Gaussian center MUST 使用 calibrated `target_beam`
- **AND** circular distance MUST 在 calibrated class order 中计算

#### Scenario: source power soft label 重排到 calibrated class order
- **WHEN** source-domain soft label 从 raw beam power/RSS vector 构造，且 MMW calibration 已启用
- **THEN** distribution class 维 MUST 按 raw→calibrated mapping 重排
- **AND** distribution mask 和 horizon 对齐 MUST 保持不变

#### Scenario: hard-label evaluation 仍使用 calibrated hard label
- **WHEN** validation 或 evaluation batch 同时包含 calibrated `target_beam` 和 `target_beam_distribution`
- **THEN** hard-label Top-K、DBA 和 validation/evaluation loss MUST 使用 calibrated `target_beam`
- **AND** metrics metadata MUST declare the calibrated label space
