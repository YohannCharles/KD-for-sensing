## ADDED Requirements

### Requirement: Target sensitive auxiliary supervision policy
训练 runtime MUST 对 target split 中的 sensitive supervision 字段实施显式 policy。`beam`、`beam_power`、CSI/channel、`path_params`、`path_descriptor`、`path_semantic_label` 和 `radio_semantic_label` MUST 按 split、label budget、labeled subset 状态和显式 opt-in 配置决定是否可被训练 loss 使用。

#### Scenario: unlabeled target 禁止 sensitive supervision
- **WHEN** target adaptation batch 来自 unlabeled target subset 或 `label_budget=0`
- **THEN** 训练 loss 访问真实 target `beam`、`beam_power`、CSI/channel、path 或 radio semantic 字段作为监督 MUST 失败
- **AND** error message MUST 包含 split、field name、label budget、labeled subset 状态和可执行修复提示

#### Scenario: labeled target auxiliary supervision 需要 opt-in
- **WHEN** `label_budget>0` 且 batch 来自 labeled target subset
- **THEN** 系统 MUST 允许 supervised beam loss 使用 labeled beam target
- **AND** path auxiliary supervision MUST 只有在显式启用 `allow_labeled_target_path_supervision` 或等价配置时才能使用
- **AND** radio auxiliary supervision MUST 只有在显式启用 `allow_labeled_target_radio_supervision` 或等价配置时才能使用
- **AND** 未启用 opt-in 时访问对应字段作为训练监督 MUST 失败

#### Scenario: sensitive usage metadata 可追踪
- **WHEN** target adaptation run 完成或失败
- **THEN** run metadata MUST 记录 sensitive field policy、label budget、labeled subset 状态和每类 target sensitive 字段是否被训练使用
- **AND** metadata MUST 至少覆盖 `used_target_beam_for_training`、`used_target_beam_power_for_training`、`used_target_csi_for_training`、`used_target_path_params_for_training`、`used_target_path_label_for_training` 和 `used_target_radio_label_for_training`
- **AND** 这些字段 MUST 可被下游 summary 和 quick conclusion 消费
