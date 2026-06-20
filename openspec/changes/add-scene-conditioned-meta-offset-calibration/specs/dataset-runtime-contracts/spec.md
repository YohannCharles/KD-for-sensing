## ADDED Requirements

### Requirement: Synthetic scenario-hyperbeam dataset runtime
系统 MUST 提供 synthetic scenario-hyperbeam dataset descriptor、sample index、modality adapter 或等价 runtime 组合，用于无真实数据的 scene-conditioned meta-offset smoke。该 dataset MUST 输出 flat sample，并 MUST 记录 scene/town/scenario/weather/domain、target_state、object_tokens、scene_params、beam target、可选 beam_power/angle/auxiliary targets 和 metadata。

#### Scenario: synthetic sample 字段完整
- **WHEN** 从 synthetic scenario-hyperbeam dataset 取一个样本
- **THEN** sample MUST 至少包含启用 sensing modalities 对应字段、`target_state`、`scene_params`、`scene_id`、`town_id`、`scenario_id`、`weather_id`、`beam_label` 和 `metadata`
- **AND** 未启用或不可用的模态 MUST 按既有 flat sample 语义缺省或标记 unavailable，而不是读取真实资源

#### Scenario: episode support query split
- **WHEN** episodic sampler 构造一个 scene task episode
- **THEN** episode MUST 包含 support subset、query subset、domain key、K-shot 数、support label policy 和 split seed
- **AND** support 和 query sample ids MUST 无交集
- **AND** K=0 时 support labels MUST 不作为训练监督暴露

#### Scenario: target sensitive guard 覆盖 support/query
- **WHEN** episode 来自 target adaptation split 且 support 或 query subset 为 unlabeled/label_budget=0
- **THEN** runtime guard MUST 禁止 loss 读取 beam、beam_power、CSI/channel、path 或 radio supervision 字段
- **AND** run metadata MUST 记录每类 sensitive 字段是否被训练使用
