## ADDED Requirements

### Requirement: Physics-informed baseline eligibility boundary
系统 MUST 区分 MMW sensor-assisted 主结论 run 与 physics-informed research/diagnostic run。任何将当前完整 CSI/channel、path params、beam power、beamspace power 或 radio/path semantic label 作为模型输入或 target-side 训练监督的 run MUST 在 metadata 中标记 sensitive usage，并且 MUST 不自动进入 sensor-assisted 主结论集合。受限 CSI 输入 MUST 明确标记为 `csi_observed` profile，而不是当前完整 CSI。

#### Scenario: 当前完整 CSI 作为输入只能是 oracle
- **WHEN** physics-informed 配置启用 `csi_input_mode=oracle_full`
- **THEN** run metadata MUST 设置 `used_csi_as_input=true`
- **AND** run metadata MUST 设置 `used_current_full_csi_as_input=true`
- **AND** `main_conclusion_eligible` MUST 为 false
- **AND** summary MUST 保留该 run 作为 oracle upper-bound baseline 或 supplementary result

#### Scenario: 受限 CSI 输入可审计
- **WHEN** physics-informed 配置启用 `csi_input_mode=history`、`partial`、`noisy` 或 `compressed`
- **THEN** run metadata MUST 记录 `csi_input_mode`
- **AND** summary MUST 明确该输入为受限 `csi_observed` 或历史 CSI
- **AND** 系统 MUST 不把当前完整 `csi_target` 传入模型 forward

#### Scenario: target path 监督排除主结论
- **WHEN** target adaptation 使用 target-side path params 或 path descriptor 计算训练 loss
- **THEN** metadata MUST 设置 `used_target_path_label_for_training=true`
- **AND** summary MUST 记录 exclusion reason
- **AND** 该 run MUST 不进入 sensor-assisted adapted-vs-source 主结论比较

#### Scenario: source-only 物理监督不污染 target eligibility
- **WHEN** physics-informed run 只在 source split 使用 CSI、path 或 beam power 物理监督
- **THEN** metadata MUST 记录 source supervision fields
- **AND** target-side sensitive usage flags MUST 保持 false
- **AND** summary MUST 仍明确该 run 的 profile 为 physics-informed 而不是纯 sensor-assisted input profile
