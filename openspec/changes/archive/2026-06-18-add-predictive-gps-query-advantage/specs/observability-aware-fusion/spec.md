## ADDED Requirements

### Requirement: Reliability-aware predictive gate inputs
Observability-aware fusion MUST support opt-in predictive gates that fuse current image latent, temporal predicted latent and GPS-derived residual latent using continuous reliability signals. The gate MUST not require these signals for existing non-predictive baselines.

#### Scenario: Gate 消费连续 reliability fields
- **WHEN** Predictive GPS-query++ enables reliability-aware gate
- **THEN** gate MAY consume `image_valid_mask`、`image_observability_score`、`image_current_missing_mask`、`gps_valid_mask`、`gps_counterfactual_mask`、`gps_delay_steps` and latent consistency scores
- **AND** missing optional fields MUST either use configured fallback values or produce clear warnings

#### Scenario: 普通 baseline 不要求 reliability fields
- **WHEN** Image ResNet+GPS, mean-pooling JEPA or existing GPS-query baseline runs without predictive gate enabled
- **THEN** model forward MUST NOT require new reliability fields
- **AND** existing training/evaluation configs MUST remain runnable

### Requirement: Condition id isolation for predictive gates
Predictive gates MUST NOT directly consume benchmark condition identifiers. Condition identifiers MAY be recorded for diagnostics and aggregation only.

#### Scenario: Gate 输入不包含 condition id
- **WHEN** Predictive GPS-query++ forward receives benchmark metadata containing `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx` or `d_idx`
- **THEN** gate input tensor MUST exclude those fields
- **AND** diagnostics MUST record `condition_id_consumed=false`

#### Scenario: Condition id 可用于 report 分组
- **WHEN** evaluation aggregates diagnostics by condition
- **THEN** reports MAY use condition ids for grouping, filenames and summary tables
- **AND** this grouping MUST occur outside model forward or gate input construction

### Requirement: Predictive branch weight diagnostics
Reliability-aware fusion MUST report how much current image, temporal predicted latent and GPS residual branches contribute to the fused representation.

#### Scenario: 输出 branch weights
- **WHEN** predictive gate forward succeeds
- **THEN** diagnostics MUST include current image weight、temporal predicted latent weight、GPS residual weight or equivalent normalized branch scores
- **AND** diagnostics MUST include batch/time aggregation suitable for per-condition reports

#### Scenario: 低 image observability 提高 predicted latent 使用
- **WHEN** image observability is low and temporal predicted latent is available
- **THEN** gate MUST be capable of increasing predicted latent branch weight according to learned or configured reliability logic
- **AND** diagnostics MUST record whether predicted branch was available and selected more strongly than in clean reference conditions

#### Scenario: wrong GPS 降低 GPS residual 使用
- **WHEN** `gps_counterfactual_mask` is true or GPS reliability score is low
- **THEN** gate MUST be capable of reducing GPS residual branch weight
- **AND** diagnostics MUST record the reliability signal that caused the reduction
