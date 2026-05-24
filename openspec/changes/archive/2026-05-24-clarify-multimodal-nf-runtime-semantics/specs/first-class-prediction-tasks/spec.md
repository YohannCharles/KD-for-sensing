## ADDED Requirements

### Requirement: Near-field beam selection 产物语义
系统 MUST 将 `near_field_beam_selection` 作为 Multimodal-NF 近场 codebook beam target 的一等 objective 语义记录在 runtime metadata、metrics metadata 和 final config 中。该 objective MUST 与 Raymobtime `current_beam_selection` 区分。

#### Scenario: near-field objective metadata
- **WHEN** 当前 objective 为 `near_field_beam_selection`
- **THEN** objective metadata MUST 声明默认主指标、metric mode、可用 beam Top-K 指标和 history/TensorBoard 字段
- **AND** runtime metadata MUST 记录 target schema 为 near-field codebook beam selection

#### Scenario: 区分 Raymobtime current beam
- **WHEN** 用户比较 `near_field_beam_selection` 和 `current_beam_selection` run
- **THEN** 两类 run 的 runtime metadata MUST 能通过 dataset type、task semantics 和 target schema 区分
- **AND** 系统 MUST 不把 Multimodal-NF near-field run 误标为 Raymobtime current snapshot run

### Requirement: Selection multitask target metadata
`selection_multitask` objective MUST 在 runtime metadata 中记录 beam selection、LOS classification 和 link quality regression 三个 target 的启用状态、loss 字段和 metric 字段。

#### Scenario: Multitask metadata 完整
- **WHEN** 当前 objective 为 `selection_multitask`
- **THEN** runtime metadata MUST 记录启用 targets、主 metric、metric mode、每个 target 的 output/head 名称和关键 loss 字段
- **AND** metrics output MUST 能追溯 beam、LOS 和 link 三类指标
