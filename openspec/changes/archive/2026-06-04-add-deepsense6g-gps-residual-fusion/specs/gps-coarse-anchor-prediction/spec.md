## ADDED Requirements

### Requirement: DeepSense6G GPS v2 prior artifact export
GPS v2 workflow MUST support exporting beam-level prior logits and an index file for downstream residual correction without changing existing prediction semantics.

#### Scenario: 保存 GPS v2 logits
- **WHEN** 用户显式启用 GPS v2 `save_logits`
- **THEN** 系统 MUST 写出 `gps_logits.npy`，形状为 `[N, 64]`
- **AND** 系统 MUST 写出 `gps_logits_index.csv`，包含 scene、sample id 和 row index
- **AND** 当配置启用 probability export 时，系统 MUST 写出 `gps_prior_probs.npy`
- **AND** predictions 与 summary 的既有字段语义 MUST 保持兼容

#### Scenario: logits index 可追踪
- **WHEN** downstream residual manifest 读取 GPS logits
- **THEN** `gps_logits_index.csv` MUST 能把每个 logits row 映射回 scene 与 sample id
- **AND** index 中重复或缺失映射 MUST 被清晰拒绝

### Requirement: DeepSense6G GPS prior fallback
当 GPS v2 logits 不可用时，下游 residual workflow MUST 能从 GPS top1 prediction 构造 circular Gaussian fallback prior，并显式记录 fallback 来源。

#### Scenario: fallback Gaussian 不使用 target label
- **WHEN** residual workflow 使用 fallback Gaussian prior
- **THEN** prior center MUST 来自 GPS predicted top1
- **AND** prior MUST NOT 使用 target label、query label 或 beam power oracle
- **AND** metadata MUST 记录 `gps_prior_source=fallback_gaussian_from_top1`

#### Scenario: fallback sigma 可配置
- **WHEN** 用户设置 `residual.gps_prior_fallback_sigma`
- **THEN** fallback prior MUST 使用该 sigma 生成 circular Gaussian logits 或 probability
- **AND** 默认 sigma MUST 为 `2.0`
