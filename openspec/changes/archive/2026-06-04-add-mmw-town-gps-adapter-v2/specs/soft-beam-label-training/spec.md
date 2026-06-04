## ADDED Requirements

### Requirement: Circular soft target loss for MMW Town GPS v2
系统 MUST 为 MMW Town GPS-only v2 提供 circular soft target supervised loss。soft target MUST 使用 circular beam distance 构造 Gaussian distribution，并 MUST 归一化为概率分布。

#### Scenario: circular soft target wrap-around
- **WHEN** `num_beams=64` 且 target beam 为 `0`
- **THEN** circular soft target MUST 将 beam `63` 作为距离 1 的邻居
- **AND** distribution 的概率和 MUST 在数值容差内等于 1

#### Scenario: circular soft CE 参与训练
- **WHEN** v2 配置 `loss.type: circular_soft_ce`
- **THEN** 系统 MUST 使用 circular soft target 计算 supervised beam loss
- **AND** validation/evaluation metrics MUST 继续使用 hard target label

### Requirement: Focal and class-balanced circular soft loss
系统 MUST 支持在 circular soft CE 上启用 focal gamma 和 class-balanced weighting。class weight 模式 MUST 至少支持 `none`、`inverse_freq`、`inverse_sqrt_freq` 和 `effective_num`，并 MUST 从当前训练 split 的 label histogram 计算。

#### Scenario: class weight 默认关闭
- **WHEN** v2 配置未显式启用 class-balanced weighting
- **THEN** 系统 MUST 使用 `class_weight: none`
- **AND** weighted ablation MUST NOT 污染 unweighted ablation 的 summary

#### Scenario: effective_num 权重记录元数据
- **WHEN** v2 配置 `loss.class_weight: effective_num`
- **THEN** 系统 MUST 使用配置的 beta 计算 class weights
- **AND** run metadata MUST 记录 beta、label histogram、权重归一化策略和 fit split

### Requirement: Circular soft loss is not KD
MMW Town GPS v2 的 circular soft CE、focal circular soft CE 和 class-balanced circular loss MUST 作为 supervised beam smoothing loss 记录，不得标记为 teacher-student KD。

#### Scenario: loss 日志使用非 KD 命名
- **WHEN** v2 训练使用 circular soft loss
- **THEN** loss diagnostics MUST 使用 `loss/beam_circular_soft_ce`、`loss/beam_focal_circular_soft_ce` 或等价非 KD 命名
- **AND** diagnostics MUST NOT 生成新的 `loss/distillation` 或 `loss/kd_soft_label` 字段表示该 supervised loss
