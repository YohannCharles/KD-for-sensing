## ADDED Requirements

### Requirement: Path auxiliary target flat sample 契约
RuntimeDataset 或等价 dataset MUST 支持在 flat sample 中表达 path-level auxiliary targets。该契约 MUST 将 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid` 标记为 target/diagnostic 字段，而不是 input modality 字段。

#### Scenario: flat sample 包含 path auxiliary targets
- **WHEN** 当前 dataset family 支持 path-level propagation parameters 且配置启用 path semantics
- **THEN** flat sample MAY 包含 `path_params`、`path_descriptor`、`path_semantic_label` 和 `path_valid`
- **AND** sample metadata MUST 保留 sample_id、dataset family、split、domain、town/scenario/weather 或等价 domain fields

#### Scenario: enabled modalities 不包含 path 或 CSI
- **WHEN** dataloader 根据配置解析 enabled modalities
- **THEN** channel、CSI、path_params 和 path_descriptor MUST NOT 被加入模型输入模态列表
- **AND** runtime metadata MUST 将这些字段记录为 auxiliary target、diagnostic 或 unavailable，而不是 sensing input

### Requirement: Unlabeled target sensitive field guard
训练 runtime MUST 提供 batch 级 sensitive field guard 或等价机制，用于阻止 label_budget 为 0 或 unlabeled target batch 的训练 loss 访问真实 target supervision 字段。

#### Scenario: label_budget 为 0 时访问敏感字段失败
- **WHEN** target adaptation 的 `label_budget=0` 且 loss 代码尝试读取 beam、beam_power、CSI/channel、path_params、path_descriptor、path_semantic_label 或 radio_semantic_label 作为训练监督
- **THEN** 系统 MUST raise error
- **AND** error message MUST 包含 split、field name、label budget 和可执行修复提示

#### Scenario: adapt log 记录防泄漏标志
- **WHEN** target adaptation run 完成或失败
- **THEN** `adapt_log.json` 或等价 metadata MUST 记录 `used_target_beam_for_training`、`used_target_beam_power_for_training`、`used_target_csi_for_training`、`used_target_path_params_for_training`、`used_target_path_label_for_training` 和 `used_target_radio_label_for_training`
- **AND** `label_budget=0` 的成功 run 中这些字段 MUST 为 false

#### Scenario: labeled target subset 可显式使用 path supervision
- **WHEN** `label_budget>0` 且 batch 来自 labeled target subset
- **THEN** training runtime MAY 允许 supervised beam loss
- **AND** 只有 `allow_labeled_target_path_supervision=true` 时，runtime MAY 允许 path_semantic_label 或 path_descriptor supervision
- **AND** unlabeled target subset MUST 继续触发 sensitive field guard
