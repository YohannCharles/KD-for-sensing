## ADDED Requirements

### Requirement: Visual-ambiguous hard negative condition
The modality difficulty pipeline MUST support a visual-ambiguous hard negative condition for GPS-query evaluation. This condition MUST preserve target labels and sample identity while selecting or marking peer examples whose visual context is similar but beam target differs by a configured margin.

#### Scenario: 构造视觉歧义 peer
- **WHEN** a difficulty profile enables visual-ambiguous hard negative selection
- **THEN** system MUST select peer samples using same split/scene constraints and configured visual similarity proxy or embedding source
- **AND** selected peers MUST satisfy configured target beam offset threshold unless fallback is recorded
- **AND** metadata MUST record source sample id、scene、similarity score、beam offset、seed and fallback reason

#### Scenario: 不改变监督 target
- **WHEN** visual-ambiguous hard negative condition is applied
- **THEN** `target_beam`、`beam_power`、sample id and split metadata MUST remain unchanged
- **AND** any peer feature substitution or metadata marking MUST be recorded as counterfactual input intervention

### Requirement: Beam-offset-constrained wrong GPS
The pipeline MUST support wrong-GPS replacement constrained by beam offset, so plausible wrong GPS interventions are strong enough to test GPS-query reliance.

#### Scenario: wrong GPS 满足 beam offset 下限
- **WHEN** profile enables beam-offset-constrained wrong GPS
- **THEN** replacement GPS MUST be selected from a peer sample whose target beam differs by at least the configured threshold
- **AND** metadata MUST record peer sample id、beam offset、GPS distance、selection pool size and scene/split constraint

#### Scenario: peer 不足时 fallback 可审计
- **WHEN** no peer satisfies the beam offset and scene/split constraints
- **THEN** system MUST use configured fallback, skip, or fail behavior
- **AND** warnings MUST record affected sample count、fallback mode、threshold and available pool size

### Requirement: Combined GPS-query advantage perturbations
The pipeline MUST support combined perturbations that pair GPS reliability degradation with image observability degradation for GPS-query advantage evaluation.

#### Scenario: CxD advantage condition 应用
- **WHEN** a profile requests `C3_random_async` or `C4_severe_async` combined with `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing` or `D7_joint_worst_case`
- **THEN** system MUST apply both GPS and image operators in a deterministic order
- **AND** metadata MUST include both GPS condition and image condition parameters, but model gate inputs MUST receive only continuous reliability fields and masks

#### Scenario: combined perturbation 保持 no-future-leak
- **WHEN** combined perturbation is applied to a temporal sequence
- **THEN** any history source used for temporal prediction or fallback MUST be strictly earlier than the prediction step
- **AND** replay metadata MUST expose history source ranges for audit

### Requirement: Advantage difficulty determinism
GPS-query advantage conditions MUST be deterministic under the same seed, split, sample id, condition id and operator parameters.

#### Scenario: 同 seed 重放一致
- **WHEN** tests apply the same advantage difficulty profile twice to the same synthetic batch
- **THEN** image/GPS tensors, masks, source indices, selected peer ids and replay metadata MUST match exactly

#### Scenario: 不同 seed 可改变 peer 但保留约束
- **WHEN** tests apply the same profile with a different seed
- **THEN** peer selection MAY differ
- **AND** all configured beam offset, scene/split and no-future-leak constraints MUST still hold
