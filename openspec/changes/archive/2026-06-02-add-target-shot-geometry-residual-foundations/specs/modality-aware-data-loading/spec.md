## ADDED Requirements

### Requirement: Geometry-residual label 字段按需加载
数据加载流程 MUST 在 `label_space.type: geometry_residual` 时按需加载或派生 geometry-residual label 字段。未启用 geometry-residual label_space 时，dataset MUST 不要求 position/geometry 字段，也不得改变现有 absolute beam label sample keys。

#### Scenario: geometry_residual 启用时返回新增标签字段
- **WHEN** 用户运行启用 `label_space.type: geometry_residual` 的配置
- **THEN** dataset 或 target provider MUST 返回 absolute beam label、geometry coarse beam 和 residual label 可用字段
- **AND** batch preparation MUST 保持这些字段 shape 可默认 collate

#### Scenario: 默认 absolute 配置不读取 geometry
- **WHEN** 用户运行现有 absolute beam classifier 配置
- **THEN** dataset MUST 继续按启用模态读取 sensing input 和 absolute beam label
- **AND** 缺少 GPS/pose/relative geometry 不得阻止 dataset 构建

### Requirement: target-shot split 字段隔离
数据加载流程 MUST 根据 split artifact 中的 subset 标记构建 source、target_labeled、target_unlabeled 和 target_test dataloader。target_unlabeled loader MUST 能提供 sensing input 和非监督 metadata，但训练 payload MUST 不暴露可作为监督的 target labels。

#### Scenario: target_unlabeled loader 隔离监督字段
- **WHEN** 构建 target_unlabeled adaptation loader
- **THEN** batch metadata MUST 标记 subset 为 `target_unlabeled`
- **AND** training payload MUST 不允许 loss 访问 beam/residual supervision 字段

#### Scenario: target_test loader 只用于评估
- **WHEN** 构建 target_test loader
- **THEN** batch MAY 包含 evaluation metrics 所需 label
- **AND** run metadata MUST 标记 target_test labels 只可在 evaluation scope 使用
