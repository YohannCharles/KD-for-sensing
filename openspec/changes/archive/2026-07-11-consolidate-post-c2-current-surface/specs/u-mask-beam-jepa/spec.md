## ADDED Requirements

### Requirement: S1-S4 temporal-router 不属于 current U-Mask contract
U-MaskBeamJEPA current contract MUST 不要求 `s1_temporalagg_modality`、`s2_pertime_modality`、`s3_two_level`、`s4_global`、temporal router distillation 或其 oracle diagnostics。删除这些提前实现 MUST 保持 `pcpg`、`bprr`、`raw_conf_gate`、`weighted_sum`、`concat_mlp`、`supervised_router`、`reliability_biased_missing_attention`、beam prototype alignment、full-to-partial teacher stabilization 和现有 U-Mask loss/config 行为。

#### Scenario: S1-S4 config 被拒绝
- **WHEN** current config 请求 S1-S4 temporal router type
- **THEN** model/config validation MUST reject the value
- **AND** 系统 MUST 不静默映射到 protected fusion branch

#### Scenario: Protected branches 保持可用
- **WHEN** focused tests 构建既有 protected fusion/loss branches
- **THEN** model forward、metadata 和 loss MUST 保持本 change 前语义
- **AND** 删除 S1-S4 MUST 不改变默认 `temporal_router_type=none` 等价行为

#### Scenario: RBMA/prototype token 不触发误删
- **WHEN** cleanup 折叠旧 `rbma-prototype-kd-missing-workflow` tombstone
- **THEN** current U-Mask RBMA fusion、prototype alignment、full-to-partial teacher 和 pattern-balanced metrics MUST 保留
- **AND** tests MUST 继续覆盖其 enabled/disabled paths
