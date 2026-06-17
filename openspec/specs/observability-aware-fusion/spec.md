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

### Requirement: JEPA advantage condition
系统 MUST 支持显式 JEPA advantage condition：当 GPS 为 `C3_random_async` 或 `C4_severe_async`，且 image 为 `D3_motion_blur`、`D4_partial_occlusion`、`D6_burst_missing` 或 `D7_joint_worst_case` 时，Image-JEPA+GPS 配置 MUST 能更多依赖 temporal latent prediction，而不是 raw CNN feature。

#### Scenario: 命中 JEPA advantage condition
- **WHEN** benchmark condition 为 `C3_random_async + D4_partial_occlusion`
- **THEN** Image-JEPA+GPS 模型 MUST 将 condition metadata 传入 fusion/gating
- **AND** observability-aware fusion MUST 标记 `jepa_advantage_condition=true`
- **AND** diagnostics MUST 记录 temporal latent prediction 权重或 fallback 状态

#### Scenario: clean condition 不强制 fallback
- **WHEN** benchmark condition 为 `C0_sync + D0_full_image`
- **THEN** observability-aware fusion MUST 不因 Scenario D 配置强制启用 JEPA fallback
- **AND** Image ResNet+GPS 或 standard fusion baseline MUST 仍可作为 clean ceiling 对照
