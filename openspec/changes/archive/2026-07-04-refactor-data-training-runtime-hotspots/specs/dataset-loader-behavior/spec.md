## ADDED Requirements

### Requirement: Dataset hotspot 拆分必须保持 loader 行为
Dataset 重构 MUST 保持 lazy loading、enabled modality resolution、sample cache behavior、scaler fitting、no-future-leak target construction 和 run metadata。

#### Scenario: 拆分后样本契约兼容
- **WHEN** DeepSense6GDataset or MMWDataset helper boundaries change
- **THEN** existing sample keys, target tensors, auxiliary target metadata, cache metadata and warning behavior MUST remain compatible
- **AND** focused tests MUST 不要求真实数据文件
