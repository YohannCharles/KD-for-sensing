# observability-aware-fusion Specification

## Purpose
定义 observability-aware fusion 如何消费 image/GPS reliability metadata、计算 modality weights、执行 adaptive fusion，并在 image degraded 或 missing 时通过 uncertainty gating 使用 JEPA temporal latent fallback。
## Requirements
### Requirement: Observability-aware fusion 输入契约
系统 MUST 提供 observability-aware fusion 模块，用于消费 image/GPS 表征和输入 reliability metadata。模块 MUST 接收 `z_img`、`z_gps`、`image_valid_mask`、`image_observability_score`、`gps_valid_mask`、`gps_delay_steps` 或 manifest 声明的等价字段，并 MUST 输出 fused representation 和可诊断的 modality weights。

#### Scenario: 构建 observability-aware fusion
- **WHEN** 配置声明使用 observability-aware fusion
- **THEN** 系统 MUST 构建接收 image 与 GPS latent 的 fusion 模块
- **AND** 模块 MUST 校验 image/GPS latent 的 batch/time 维兼容
- **AND** 缺失必需 reliability metadata 时 MUST 抛出清晰错误或使用配置声明的 fallback 并记录 warning

#### Scenario: 输出 modality weights
- **WHEN** observability-aware fusion forward 成功
- **THEN** 输出 MUST 包含 fused latent 或 logits 可消费的 representation
- **AND** diagnostics MUST 包含 `w_img`、`w_gps`、`image_observability_score` 和 GPS reliability summary

### Requirement: Reliability weighting 和 adaptive fusion
Observability-aware fusion MUST 根据 image observability 与 GPS reliability 计算 modality weights，并执行 adaptive fusion。默认 fusion 语义 MUST 等价于 `z_fuse = w_img * z_img + w_gps * z_gps` 或在维度不一致时使用显式 projection 后的等价加权融合。

#### Scenario: GPS async 降低 GPS 权重
- **WHEN** `gps_valid_mask` 显示 GPS missing 或 `gps_delay_steps` 较高
- **THEN** fusion MUST 降低 GPS reliability score 或 `w_gps`
- **AND** diagnostics MUST 记录降低权重的原因字段

#### Scenario: 图像缺失降低 image 权重
- **WHEN** `image_valid_mask` 为 false 或 `image_observability_score` 低于阈值
- **THEN** fusion MUST 降低 image reliability score 或 `w_img`
- **AND** physical corruption 与 missing MUST 在 diagnostics 中可区分

### Requirement: 不确定性 gating 与 JEPA fallback
Observability-aware fusion MUST 支持 uncertainty gating。当 image observability 低于阈值且 JEPA temporal prediction 可用时，系统 MUST 能选择 temporal JEPA latent prediction 作为 degraded/missing image 的 fallback，而不是强制使用 raw CNN features。

#### Scenario: 低可观测性触发 JEPA fallback
- **WHEN** `image_observability_score` 低于配置阈值且模型提供 temporal JEPA predicted latent
- **THEN** fusion MUST 使用 predicted latent 或提高其权重
- **AND** diagnostics MUST 记录 fallback 是否触发、触发阈值和使用的 latent source

#### Scenario: fallback 不可用时降级
- **WHEN** 配置启用 JEPA fallback 但当前模型不提供 temporal predicted latent
- **THEN** 系统 MUST 使用配置声明的 fallback 策略
- **AND** 系统 MUST 在 warnings 中记录 unavailable reason

### Requirement: Proto-compatible reliability mask weighted fusion
The system MUST support a lightweight reliability mask weighted fusion option that is compatible with prototype prediction and randomdrop subset training. Missing modalities MUST receive zero weight and available modality weights MUST be normalized over available modalities only.

#### Scenario: missing modality receives zero weight
- **WHEN** reliability fusion receives modality features and an availability mask
- **THEN** every unavailable modality MUST have weight zero within numerical tolerance
- **AND** unavailable modality features MUST NOT contribute to the fused representation

#### Scenario: available weights normalize
- **WHEN** at least one modality is available for a sample
- **THEN** reliability weights over available modalities MUST sum to one within numerical tolerance
- **AND** the fused representation MUST be equivalent to a weighted sum over available modality features

#### Scenario: lightweight implementation boundary
- **WHEN** reliability fusion is enabled for Scene31 subset candidates
- **THEN** the implementation MUST use a small scorer such as pooled feature plus availability or learned modality reliability embeddings
- **AND** it MUST NOT introduce a complex transformer, imputation module or external dependency

#### Scenario: epoch-level reliability log
- **WHEN** training with reliability fusion completes an epoch
- **THEN** the run directory MUST contain or support writing `reliability_weights_epoch.csv`
- **AND** rows MUST include epoch, pattern, modality, mean_weight, std_weight and available_rate

