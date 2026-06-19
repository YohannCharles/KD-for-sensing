## ADDED Requirements

### Requirement: Logit-level uncertainty fusion
Observability-aware fusion MUST 支持 opt-in 的 logit-level uncertainty/evidence fusion 模式，用于 geometry-prior 模型。该模式 MUST 使用 reliability、uncertainty 或 evidence 信号组合各 branch logits，同时 MUST 不要求普通 baseline 消费这些字段。

#### Scenario: branch uncertainty controls weights
- **WHEN** geometry-prior logit fusion receives image logits、geometry prior logits 和 branch entropy/evidence
- **THEN** fusion MAY reduce the weight of high-uncertainty or low-reliability branches
- **AND** diagnostics MUST record branch entropy/evidence、final weights and unavailable reason

#### Scenario: ordinary baseline ignores metadata
- **WHEN** Image ResNet+GPS、JEPA GPS-query k=4 或其它未 opt-in 的 baseline 在同一 benchmark batch 上运行
- **THEN** reliability、uncertainty 和 branch diagnostic fields MUST NOT be required forward inputs
- **AND** batch runtime MUST allow those models to ignore unsupported metadata

### Requirement: Geometry-prior condition id isolation
Geometry-prior reliability fusion MUST NOT consume benchmark condition identifiers as model inputs. Condition identifiers MAY only be used outside model forward for aggregation, filenames and reports.

#### Scenario: condition id 不进入 fusion input
- **WHEN** batch metadata contains `condition`、`predictive_condition_id`、`gps_condition`、`image_condition`、`c_idx` or `d_idx`
- **THEN** logit fusion input tensor MUST exclude those fields
- **AND** diagnostics MUST record `condition_id_consumed=false`

#### Scenario: condition id 用于 report 分组
- **WHEN** evaluation aggregates P0-P5 or advantage metrics
- **THEN** reports MAY use condition ids for grouping and table labels
- **AND** this grouping MUST happen outside model forward and gate/fusion input construction

### Requirement: GPS reliability in logit fusion
Geometry-prior logit fusion MUST be capable of down-weighting GPS prior when GPS reliability metadata or branch disagreement indicates likely wrong, delayed or unavailable GPS.

#### Scenario: wrong GPS 降低 prior 权重
- **WHEN** `gps_counterfactual_mask=true` or prior-image disagreement exceeds configured threshold
- **THEN** fusion MUST be capable of reducing geometry prior weight
- **AND** diagnostics MUST record the reliability signal or disagreement metric used

#### Scenario: clean high-agreement GPS 可提高 prior 权重
- **WHEN** GPS is valid, delay is low, prior entropy is low and prior-image agreement is high
- **THEN** fusion MAY increase geometry prior weight
- **AND** diagnostics MUST compare clean weight distribution against hard-condition weight distribution

### Requirement: Image observability in logit fusion
Geometry-prior logit fusion MUST be capable of down-weighting image logits when image observability is low, while still protecting clean performance.

#### Scenario: image degradation 降低 image 权重
- **WHEN** `image_valid_mask=false` or `image_observability_score` is below configured threshold
- **THEN** fusion MUST be capable of lowering image branch weight or increasing uncertainty
- **AND** diagnostics MUST distinguish missing image, occlusion, blur and burst missing where metadata is available

#### Scenario: clean condition 不强制降低 image 权重
- **WHEN** condition is clean and image observability is high
- **THEN** fusion MUST NOT force a low image weight solely because geometry prior is enabled
- **AND** clean branch weights MUST be reported as part of clean regression diagnostics
