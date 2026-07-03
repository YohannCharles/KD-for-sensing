## ADDED Requirements

### Requirement: Local baseline stress comparability
本地缺失模态 baseline MUST 能声明 stress benchmark comparability metadata。适用对象包括 AMBER-lite、AMBER full、RMBP-MM、U-MaskBeamJEPA/weighted_sum 和其它 current local missing-modality baseline。

#### Scenario: baseline 声明 required fields
- **WHEN** baseline 被纳入 missing-modality stress suite
- **THEN** baseline manifest 或 run metadata MUST 声明 config path、weights path、checkpoint provenance、modalities、split、sample_count、label_space、metric_profile、target_source、seed 和 difficulty_digest
- **AND** 缺失 required field MUST 阻止该 baseline 进入 strict claim comparison

#### Scenario: local substitute 状态保留
- **WHEN** AMBER、RMBP-MM 或其它外部论文 baseline 使用本仓库 local implementation
- **THEN** stress summary MUST 保留 `local experimental baseline` 或 `local substitute` 状态
- **AND** 系统 MUST 不将其描述为 official reproduction

#### Scenario: baseline 缺某模态
- **WHEN** stress suite 包含 baseline 不支持的模态或 condition
- **THEN** 对应 row MUST 标记为 unavailable、not_applicable 或 not_comparable
- **AND** summary MUST 不把缺失 row 当作 0 分或 clean 结果
