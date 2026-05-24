## ADDED Requirements

### Requirement: Dataset runtime capability purpose 明确
`dataset-runtime-contracts` spec MUST 使用真实目的说明描述 dataset descriptor、sample index、modality adapter、target provider 和 runtime metadata 契约。该 spec MUST 不长期保留 archived TBD Purpose 文案。

#### Scenario: dataset runtime purpose 不再是 TBD
- **WHEN** 开发者阅读 `openspec/specs/dataset-runtime-contracts/spec.md`
- **THEN** Purpose MUST 描述 dataset runtime contract 的当前职责
- **AND** Purpose MUST NOT 包含 `TBD - created by archiving`

### Requirement: Runtime metadata 区分 dataset family 与 target schema
Dataset runtime metadata MUST 同时记录 dataset family 信息和当前 objective target schema。dataset family MUST 表达数据来源、storage kind、split 和 profiles；target schema MUST 表达当前 run 实际训练或评估的主 target 和辅助 target。

#### Scenario: Multimodal-NF metadata 双层记录
- **WHEN** 训练或评估构建 Multimodal-NF dataloaders
- **THEN** runtime metadata MUST 记录 `dataset_type: multimodal_nf`、storage kind、split strategy、enabled modalities 和 input profiles
- **AND** runtime metadata MUST 记录当前 objective 对应的 target schema
- **AND** 二者 MUST 不互相覆盖

#### Scenario: Raymobtime 与 Multimodal-NF 语义隔离
- **WHEN** 系统写出 Raymobtime s008 和 Multimodal-NF run metadata
- **THEN** Raymobtime current snapshot beam selection MUST 使用 Raymobtime task semantics
- **AND** Multimodal-NF near-field beam selection MUST 使用近场 codebook task semantics
- **AND** 两者 MUST 不共享会导致误读的 target schema 名称
