## ADDED Requirements

### Requirement: hard pattern CE reweight
训练 runtime MUST 支持 sample-wise hard pattern loss weight。启用 `use_pattern_loss_weight=true` 时，系统 MUST 根据 `pattern_loss_weights` 对 CE loss 加权，默认 `apply_pattern_weight_to_ce=true` 且 `apply_pattern_weight_to_proto=false`。

#### Scenario: 只加权 CE
- **WHEN** `radar_only` 配置权重为 1.5 且 `apply_pattern_weight_to_proto=false`
- **THEN** radar_only 样本的 CE loss MUST 乘以 1.5
- **AND** prototype loss MUST 不因该 pattern weight 改变

#### Scenario: metrics 记录 sample weight
- **WHEN** pattern loss weight 启用
- **THEN** metrics MUST 记录 `ce_loss`、`weighted_ce_loss`、`proto_loss` 和 `avg_sample_weight`

### Requirement: mask-conditioned fusion adapter
模型 runtime MUST 支持 opt-in mask-conditioned adapter。启用 `use_mask_adapter=true` 时，adapter MUST 接收 available mask `[B, M]`，通过轻量 MLP 输出与 fused hidden dim 一致的 gamma/beta，并在 fusion 后 beam head 前调制 fused feature。

#### Scenario: 未启用 adapter 保持旧行为
- **WHEN** 配置未声明或设置 `use_mask_adapter=false`
- **THEN** 模型 forward 和 checkpoint shape MUST 与旧配置保持兼容

#### Scenario: adapter 参数量记录
- **WHEN** `use_mask_adapter=true`
- **THEN** startup/runtime metadata MUST 记录 adapter 参数量

### Requirement: weak-pattern KD
训练 runtime MUST 支持 opt-in weak-pattern KD。启用 `use_weak_pattern_kd=true` 时，系统 MUST 对 `kd_apply_patterns` 内样本使用 full modality same-model stopgrad teacher logits，并只对这些样本计算 KD loss。

#### Scenario: KD 只作用于指定 pattern
- **WHEN** `kd_apply_patterns=["radar_only", "lidar_only"]`
- **THEN** 只有 radar_only 和 lidar_only 样本贡献 KD loss
- **AND** eval 时 MUST 不启用 teacher branch

#### Scenario: KD diagnostics
- **WHEN** weak-pattern KD 启用
- **THEN** metrics MUST 记录 `kd_loss` 和 `kd_active_ratio`

### Requirement: lightweight latent prediction probe
训练 runtime MUST 支持 opt-in lightweight latent prediction probe。启用 `use_light_latent_pred=true` 时，系统 MUST 对指定 pattern 的 partial fused feature 预测 stopgrad full fused latent 或 prototype distribution，并只作为 auxiliary loss 使用。

#### Scenario: latent prediction 不插回 fusion
- **WHEN** latent predictor 产生 `h_pred` 或 `q_pred`
- **THEN** 预测结果 MUST 不替换 fused feature 或 beam head 输入
- **AND** eval 时 MUST 不启用 predictor

#### Scenario: latent prediction diagnostics
- **WHEN** latent prediction 启用
- **THEN** metrics MUST 记录 `latent_pred_loss` 和 `latent_pred_active_ratio`
