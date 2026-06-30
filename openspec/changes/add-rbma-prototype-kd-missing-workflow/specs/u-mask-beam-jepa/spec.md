## ADDED Requirements

### Requirement: RBMA fusion type
U-MaskBeamJEPA MUST 支持 `fusion_type: reliability_biased_missing_attention`。启用该 fusion 时，模型 MUST 将 canonical 模态 latent、missing mask、modality reliability 和可选 JEPA/global token 传入 RBMA attention，并 MUST 保留现有 `concat_mlp`、`weighted_sum` 和 `reliability_gated_cross_attention` 行为。

#### Scenario: RBMA fusion 构建
- **WHEN** U-MaskBeamJEPA 配置 `fusion_type` 为 `reliability_biased_missing_attention`
- **THEN** 模型 MUST 构建 RBMA fusion
- **AND** forward 输出 MUST 包含 `logits`
- **AND** diagnostics MUST 包含 attention weights、mask provenance 和 modality reliability summary

#### Scenario: 旧 fusion type 保持可用
- **WHEN** U-MaskBeamJEPA 配置 `fusion_type` 为 `concat_mlp`、`weighted_sum` 或 `reliability_gated_cross_attention`
- **THEN** 模型 MUST 继续构建对应旧路径
- **AND** 系统 MUST 不要求旧路径提供 RBMA-only diagnostics

### Requirement: No-JEPA prototype and KD training options
U-MaskBeamJEPA MUST 支持在 `use_jepa_loss=false` 时启用 beam prototype alignment 和 online full-to-partial teacher stabilization。关闭这些开关时，旧 U-MaskBeamJEPA loss 行为 MUST 不变。

#### Scenario: no-JEPA prototype KD forward payload
- **WHEN** `use_jepa_loss=false`、`use_beam_prototype_alignment=true` 或 `use_full_to_partial_kd=true`
- **THEN** forward 或 training extension MUST 暴露 student fused feature、可用 modality features 和 student logits
- **AND** 不得要求 Gaussian JEPA NLL 必需字段参与 loss

#### Scenario: 关闭增强回退旧损失
- **WHEN** `use_beam_prototype_alignment=false` 且 `use_full_to_partial_kd=false`
- **THEN** 总损失 MUST 回退到当前 U-MaskBeamJEPA 配置声明的 beam CE、teacher CE 和可选 JEPA loss
- **AND** metrics MUST 不生成 prototype 或 full-to-partial KD 标量

### Requirement: Pattern-balanced mask metrics
U-MaskBeamJEPA training 和 evaluation MUST 能记录 pattern-balanced missing mask 的 pattern name、sample count 和按 pattern 聚合的 top-k/loss 指标。

#### Scenario: training 记录 pattern 分布
- **WHEN** 训练配置启用 `mask_sampler: pattern_balanced`
- **THEN** 每个 epoch 的 metrics 或 logs MUST 记录 pattern 分布
- **AND** 至少 MUST 能区分 `full`、`missing_gps`、`non_gps_only`、`only_gps`、`random_0.5` 和 `random_0.75`

#### Scenario: evaluation 按 pattern 汇总
- **WHEN** evaluation 配置指定多个 missing patterns
- **THEN** 输出报告 MUST 按 pattern 汇总 top1、top5、loss 和样本数
- **AND** 汇总 MUST 不把 `missing_gps` 与 `non_gps_only` 的 pattern name 合并
