## ADDED Requirements

### Requirement: MMW 5% target-shot split artifact
MMW cross-scene adaptation protocol MUST support a 5% target-shot split artifact for scenario-level、town-level 和 weather/condition-level experiments. The artifact MUST be derived from MMW availability/manifest metadata and MUST preserve existing group-safe window leakage diagnostics.

#### Scenario: MMW scenario target-shot split
- **WHEN** MMW manifest 包含至少一个 source scenario 和一个 target scenario
- **THEN** split builder MUST 能生成 source、target_labeled、target_unlabeled 和 target_test split
- **AND** target_labeled MUST 默认占 target adaptation pool 的 5%
- **AND** split metadata MUST 保留 sample id、window overlap、guard band 和 strict eligibility diagnostics

#### Scenario: MMW weather target-shot split
- **WHEN** MMW availability 包含多个 weather/condition 且配置选择 condition-level target domain
- **THEN** split builder MUST 将 target condition 与 source condition 写入 metadata
- **AND** summary MUST 不把缺少其它 condition 的 run 声称为 weather-shift 验证

### Requirement: MMW geometry-residual label 统计
MMW cross-scene adaptation protocol MUST support geometry-residual label statistics using direct RSU-CAV relative geometry when available. The protocol MUST record whether `beam_geo` is derived from direct geometry, uniform angle quantization, codebook mapping or unavailable.

#### Scenario: MMW manifest geometry 可用
- **WHEN** MMW frame manifest 包含 direct relative azimuth 或可解析 RSU/CAV pose
- **THEN** geometry-residual label builder MUST 能生成 `beam_geo`、`beam_residual` 和 `geo_sector`
- **AND** split diagnostics MUST 写出 source/target absolute 与 residual histogram

#### Scenario: MMW geometry 不可用
- **WHEN** 某个 MMW sample 缺少 direct geometry 且配置要求 geometry residual
- **THEN** 系统 MUST 按 `label_space.geometry.required` 决定失败或标记 unavailable
- **AND** unavailable reason MUST 写入 manifest 或 diagnostics artifact

### Requirement: MMW target-shot 防 oracle 边界
MMW target-shot adaptation MUST only use `target_labeled` beam/residual labels for supervised target loss. `target_unlabeled` and `target_test` beam labels、beam_power、path fields、radio labels and channel-derived labels MUST NOT be used for adaptation threshold selection、prototype update、temperature fitting、early stopping or training loss.

#### Scenario: target_test label 不参与 calibration
- **WHEN** 后续 calibration 或 adaptation 使用 MMW target-shot split artifact
- **THEN** target_test labels MUST only be available in final evaluation scope
- **AND** eligibility audit MUST mark the run ineligible if target_test labels influence training, threshold, prototype, temperature or early stopping

#### Scenario: target_labeled residual 监督合法
- **WHEN** MMW adaptation 使用 `target_labeled` subset 且 label budget 大于 0
- **THEN** supervised beam 或 residual loss MAY use target_labeled labels
- **AND** usage metadata MUST record selected sample ids and target_label_fraction
