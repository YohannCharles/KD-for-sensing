## ADDED Requirements

### Requirement: MMWDataset family adapter 必须隔离增强职责
MMWDataset 重构 MUST 将 prepared sequence loading 与 family-specific geometry、availability、radio semantic、path semantic、physical label、beam power 和 physics supervision augmentation 保持分离。

#### Scenario: 修改 physical label 不触碰基础 loader
- **WHEN** developer changes physical label, beam power or radio semantic augmentation
- **THEN** primary changes MUST be in MMW family adapter or focused MMW helper modules
- **AND** DeepSense6G base loading 和 common target construction MUST 保持不变
