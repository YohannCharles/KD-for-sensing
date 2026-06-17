## ADDED Requirements

### Requirement: Predictive robustness benchmark suite
JEPA GPS shortcut benchmark MUST 支持 `predictive_jepa_robustness` suite。该 suite MUST 复用 shared difficulty pipeline，并 MUST 能与现有 model comparability、manifest、metrics 和 output boundary 机制共存。

#### Scenario: Manifest 引用 predictive suite
- **WHEN** benchmark manifest 声明 suite type `predictive_jepa_robustness`
- **THEN** runner MUST 标准化 predictive condition、difficulty operator 参数、seed、history window 和 output artifact plan
- **AND** runner MUST 将 image/GPS corruption 委托给 shared difficulty pipeline
- **AND** runner MUST 不维护独立平行的 corruption 实现

#### Scenario: 支持 required model groups
- **WHEN** manifest 声明 strict predictive robustness evaluation
- **THEN** runner MUST 至少支持 Image CNN+GPS、JEPA predictive hybrid、JEPA GPS-query-pool 或其它声明的 JEPA baseline model group
- **AND** report MUST 区分 supervised CNN visual encoder、JEPA encoder reuse、predictive auxiliary branch 和 feature-consistency fusion

### Requirement: Predictive regional aggregation
Benchmark MUST 为 Predictive Robustness suite 输出 regional aggregation 和 margin-vs-CNN summary。该 aggregation MUST 与现有 metrics_by_condition 和 robustness_summary 兼容，但 MUST 明确标记 predictive claim 口径。

#### Scenario: 写出 predictive summary
- **WHEN** predictive robustness benchmark 完成至少一个 strict comparable model group
- **THEN** runner MUST 写出 condition-level metrics、`predictive_dba`、`predictive_top1`、`cnn_predictive_dba`、`margin_vs_cnn_dba`、`claim_pass_5pt`、sample_count、seed 和 comparability status
- **AND** runner manifest MUST 在 `output_files` 中登记 predictive summary 文件

#### Scenario: 同时记录 overall sanity
- **WHEN** benchmark manifest 同时启用 Scenario D CxD 或 overall sanity
- **THEN** runner MUST 将 overall CxD metrics 与 predictive metrics 分字段记录
- **AND** report MUST 不用 overall CxD mean 覆盖 predictive main claim

### Requirement: Predictive claim comparability
Predictive robustness claim MUST 只在严格可比较行上计算。比较 MUST 保持同一 split、label space、metric profile、sample_count、seed、difficulty digest 和 target semantics。

#### Scenario: 不可比较时标记 unavailable
- **WHEN** JEPA predictive hybrid 与 Image CNN+GPS 的 split、sample_count、metric profile、difficulty digest 或 enabled predictive condition 不一致
- **THEN** runner MUST 不计算正式 `claim_pass_5pt`
- **AND** corresponding margin result MUST 标记为 `unavailable` 或 `not_comparable` 并记录原因

#### Scenario: mock 或 partial run 不生成真实 claim
- **WHEN** benchmark 使用 synthetic metrics、mock weights、partial required model groups 或 allow_missing_artifacts
- **THEN** runner MUST 继续输出 schema-compatible metrics
- **AND** claim status MUST 标记为 `mock/smoke`、`pending` 或 `unavailable`
