## ADDED Requirements

### Requirement: Missing-modality stress presets
Difficulty pipeline MUST 支持 missing-modality stress suite 所需的 canonical presets。Presets MUST 标准化 image/GPS/radar/LiDAR/mmWave 缺失、不可用、噪声和异步条件，并记录 replay metadata。

#### Scenario: 标准化 missing stress preset
- **WHEN** profile 引用 `missing_modality_stress` preset
- **THEN** 系统 MUST 标准化 clean、single missing、multi missing、only modality、random missing 和 unavailable modality 条件
- **AND** unknown condition MUST 被拒绝并列出可用 condition

#### Scenario: target 不变
- **WHEN** missing-modality stress preset 应用于 batch
- **THEN** 输入模态 tensor、valid mask、missing mask 或 reliability metadata MAY 改变
- **AND** `target_beam`、`beam_power`、soft target、sample id 和 split metadata MUST 保持不变

#### Scenario: replay metadata
- **WHEN** stress preset 完成一次 batch transform
- **THEN** metadata MUST 记录 profile id、condition、severity、operator params、seed、split、sample ids、fallback count 和 warnings

### Requirement: Sensing modality unavailable operators
Difficulty pipeline MUST 为 radar、LiDAR 和 mmWave 提供 unavailable/missing 表达，供缺失模态 stress suite 使用。

#### Scenario: radar unavailable
- **WHEN** profile 将 radar 标记为 unavailable
- **THEN** 系统 MUST 通过 zero-fill、mask token 或 valid mask false 表达 radar 不可用
- **AND** metadata MUST 记录 fallback 表达方式和 affected sample count

#### Scenario: LiDAR or mmWave unavailable
- **WHEN** profile 将 LiDAR 或 mmWave 标记为 unavailable
- **THEN** 系统 MUST 保持对应 tensor shape 可被模型消费
- **AND** 目标标签、GPS、image 和 split metadata MUST 不变
