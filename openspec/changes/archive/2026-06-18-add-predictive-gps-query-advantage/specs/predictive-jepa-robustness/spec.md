## ADDED Requirements

### Requirement: GPS-query advantage slice
Predictive Robustness MUST support an optional GPS-query advantage slice that evaluates whether GPS-conditioned JEPA prediction helps under visual ambiguity, beam-offset-constrained wrong GPS, and combined GPS/image reliability degradation. This slice MUST supplement, not replace, the canonical P0-P5 benchmark.

#### Scenario: Advantage slice 不替代 P0-P5
- **WHEN** benchmark manifest enables GPS-query advantage slice
- **THEN** output MUST still include canonical P0-P5 condition-level metrics when a predictive robustness claim is requested
- **AND** reports MUST label advantage slice as mechanism/diagnostic evidence rather than the primary P-suite claim

#### Scenario: Advantage slice 包含关键条件
- **WHEN** GPS-query advantage slice is normalized
- **THEN** it MUST include conditions covering visual ambiguity, beam-offset-constrained wrong GPS, and at least `C3_random_async` or `C4_severe_async` combined with one of `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing`、`D7_joint_worst_case`
- **AND** each condition MUST record seed、split、difficulty digest、operator params、fallback count and sample count

#### Scenario: Advantage slice 输出 per-condition margin
- **WHEN** advantage slice evaluation completes with strict comparable model rows
- **THEN** output MUST include per-condition DBA/Top-K metrics and margins against `Image ResNet+GPS` and current `JEPA GPS-query k=4` or configured GPS-query baseline
- **AND** missing strict comparable rows MUST mark the slice as unavailable or not-comparable

### Requirement: GPS-query++ strict comparison set
Predictive Robustness real evaluation for this workflow MUST compare Predictive GPS-query++ against Image ResNet+GPS and a matched current GPS-query baseline under the same protocol.

#### Scenario: Strict model groups are present
- **WHEN** a real Predictive GPS-query++ benchmark manifest is used for claim-oriented evaluation
- **THEN** manifest MUST include Image ResNet+GPS, current JEPA GPS-query baseline, and Predictive GPS-query++ model groups
- **AND** model groups MUST declare config path、weights path、checkpoint provenance、metric profile、split、sample count and label space

#### Scenario: 同协议字段一致
- **WHEN** strict comparison rows are aggregated
- **THEN** rows MUST share history window、GPS input/source window、prediction horizon、scene set、seed、difficulty digest、distance metric and beam label space
- **AND** any mismatch MUST prevent claim upgrade and appear in warnings

### Requirement: GPS-query++ claim gate
Predictive GPS-query++ claim status MUST require both canonical P0-P5 evidence and advantage-slice evidence. Advantage-slice improvements MAY explain mechanism but MUST NOT alone promote a claim.

#### Scenario: Claim gate 计算
- **WHEN** P0-P5 and advantage slice metrics are available for strict comparable models
- **THEN** system MUST compute P-suite margin vs Image ResNet+GPS、advantage-slice margin vs Image ResNet+GPS、advantage-slice margin vs current GPS-query baseline and claim pass flags
- **AND** primary claim pass MUST remain based on canonical predictive robustness criteria and configured margin threshold

#### Scenario: Advantage slice 单独提升不升级 claim
- **WHEN** Predictive GPS-query++ outperforms baselines on advantage slice but not on canonical P0-P5
- **THEN** report MUST describe the result as mechanism evidence or targeted advantage
- **AND** claim status MUST remain pending, partial, unavailable or not-comparable according to provenance

### Requirement: GPS-query++ diagnostics bundle
Predictive Robustness MUST provide a diagnostics bundle for GPS-query++ evaluations that explains branch usage without treating explanations as causal proof.

#### Scenario: 输出 gate 和 latent consistency diagnostics
- **WHEN** Predictive GPS-query++ evaluation emits diagnostics
- **THEN** bundle MUST include gate weight summaries、branch availability、latent consistency summaries、fallback counts and per-condition margin tables
- **AND** diagnostics MUST be linked from a machine-readable manifest

#### Scenario: 解释性图不构成 claim
- **WHEN** report includes attention, gate, t-SNE/PCA, rank CDF or latent consistency figures
- **THEN** report MUST state that these figures are explanatory diagnostics
- **AND** numeric claim MUST still be based on strict metrics and provenance
