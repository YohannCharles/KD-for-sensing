## ADDED Requirements

### Requirement: GPS-query token/readout paired ablation
JEPA visual architecture sweep MUST include a minimal paired ablation for GPS-query token output and token readout. The ablation MUST compare token readout candidates against matching mean, GPS-query frame, and legacy GPS-query token baselines under the same data, seed, checkpoint selection, metric profile, difficulty condition and output root.

#### Scenario: readout ablation 候选完整
- **WHEN** full 或 focused GPS-query readout sweep manifest 生成
- **THEN** manifest MUST include `pooler_mean`、`pooler_gps_query_k2_frame`、`pooler_gps_query_k2_tokens` and at least one explicit token readout candidate
- **AND** each candidate MUST record `variant_id`、family、pooler type、output mode、`k_queries`、readout type、representation core type、checkpoint policy and run tier
- **AND** manifest MUST NOT silently replace or rename existing `pooler_gps_query_k2_tokens`

#### Scenario: readout ablation 使用严格可比字段
- **WHEN** readout ablation 结果进入 strict ranking 或 claim gate
- **THEN** each row MUST include split、scene set、seed、history window、GPS input source window、prediction horizon、beam label space、metric profile、distance metric、normalization artifact、difficulty digest、checkpoint selection and output root
- **AND** any row missing these fields MUST be excluded from strict readout claim ranking

#### Scenario: readout gate 输出 paired delta
- **WHEN** summary 生成 GPS-query token/readout claim gate
- **THEN** gate MUST output paired delta versus `pooler_gps_query_k2_frame` and versus `pooler_mean`
- **AND** gate MUST include clean/P0 delta、P1-P5 mean delta、Scene31 delta、S31-S34 delta、S32-S34 delta and P3/P4 degradation-condition delta where available
- **AND** gate MUST record threshold、pass/fail status、missing evidence and caveats

#### Scenario: seed confirm 防止单 seed 误判
- **WHEN** seeds 17、23 and 42 are available for the same readout candidate
- **THEN** summary MUST report per-seed metrics and mean/std aggregation
- **AND** claim gate MUST indicate whether the readout improvement is directionally consistent across a majority of seeds

#### Scenario: query diagnostics 汇入 summary
- **WHEN** attention/query diagnostics are available for readout candidates
- **THEN** summary MUST include query diversity、attention entropy、effective patch count、readout weight summary and diagnostics availability fields
- **AND** missing diagnostics MUST be reported as `missing` or `unavailable` rather than causing the candidate row to disappear
