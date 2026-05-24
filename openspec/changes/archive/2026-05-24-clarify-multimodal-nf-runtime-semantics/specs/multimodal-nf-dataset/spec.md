## ADDED Requirements

### Requirement: Multimodal-NF capability purpose 明确
`multimodal-nf-dataset` spec MUST 使用真实目的说明描述当前 capability，覆盖本地数据布局、审计、HDF5 index、flat sample、profile 懒加载、近场 codebook target、辅助标签和 smoke workflow。该 spec MUST 不长期保留 archived TBD Purpose 文案。

#### Scenario: spec purpose 不再是 TBD
- **WHEN** 开发者阅读 `openspec/specs/multimodal-nf-dataset/spec.md`
- **THEN** Purpose MUST 描述 Multimodal-NF dataset capability 的当前职责
- **AND** Purpose MUST NOT 包含 `TBD - created by archiving`

### Requirement: Multimodal-NF objective runtime 语义
Multimodal-NF run metadata MUST 根据当前 objective 记录准确的 task semantics 和 target schema。系统 MUST 区分 dataset family 能力与当前 run 实际使用的 objective，不得把所有 Multimodal-NF run 都描述为同一个 future beam task。

#### Scenario: near-field beam selection runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: near_field_beam_selection`
- **THEN** runtime metadata MUST 记录 objective 为 `near_field_beam_selection`
- **AND** target schema MUST 表达 Multimodal-NF 近场三维 codebook flattened beam class
- **AND** metadata MUST 记录 codebook shape、flatten order 和 num beam classes

#### Scenario: LOS runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: current_los_classification`
- **THEN** runtime metadata MUST 记录 objective 为 `current_los_classification`
- **AND** target schema MUST 表达 LOS/NLOS binary classification
- **AND** metadata MUST 不把该 run 的主任务描述为 beam-only prediction

#### Scenario: selection multitask runtime
- **WHEN** 用户运行 `data.dataset.type: multimodal_nf` 且 `experiment.objective: selection_multitask`
- **THEN** runtime metadata MUST 记录 beam selection、LOS 和 link quality targets 均启用
- **AND** metadata MUST 记录每个 head 或 output 字段的 target 语义

### Requirement: Multimodal-NF codebook consistency
系统 MUST 校验 Multimodal-NF codebook metadata 与模型输出类别数的一致性。若配置解析出的 codebook `num_beam_classes` 与 beam head `num_classes` 不一致，系统 MUST 在启动阶段抛出清晰错误或拒绝写出自相矛盾的 final config。

#### Scenario: codebook 类别数一致
- **WHEN** Multimodal-NF dataset 解析出 codebook shape 和 `num_beam_classes`
- **THEN** 模型 beam head 的输出类别数 MUST 与 `num_beam_classes` 一致
- **AND** `final_config.yaml` 或 runtime metadata MUST 记录该一致性来源

#### Scenario: codebook 类别数不一致
- **WHEN** `data.dataset.codebook_metadata.num_beam_classes` 与模型 beam head 类别数不一致
- **THEN** 系统 MUST 抛出包含两个实际值和相关配置路径的清晰错误
- **AND** 系统 MUST 不继续启动一个会产生不可解释指标的训练 run
