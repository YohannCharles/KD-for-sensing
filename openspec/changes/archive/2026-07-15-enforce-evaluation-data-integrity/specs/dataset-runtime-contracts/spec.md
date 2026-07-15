## ADDED Requirements

### Requirement: 数据拟合统计量仅来自实际训练子集
所有依赖数据拟合的 normalization、scaler、streaming statistics 或校准 artifact MUST 只消费 resolved train dataset 的实际 leaf 和 effective train indices。validation/test MUST 只读复用训练 artifact，MUST NOT 自行拟合或从未过滤的父 dataset 获取统计量。

#### Scenario: 内部 train/validation split
- **WHEN** 单个父 dataset 被拆成 train 与 validation `Subset`
- **THEN** GPS、LiDAR、mmWave、CSI、position、occlusion 和其它已启用统计量 MUST 只从 train indices 拟合
- **AND** validation MUST 复用完全相同的已拟合 artifact

#### Scenario: Pooled dataset normalization
- **WHEN** train dataset 由多个 domain 或 scene leaf 组成
- **THEN** shared normalization MUST 从全部训练 leaf 的 effective indices 联合拟合
- **AND** per-domain normalization MUST 为每个 domain 保存明确映射和 provenance
- **AND** runtime MUST NOT 静默只取第一个 leaf 的统计量

#### Scenario: Validation 或 test 尝试拟合
- **WHEN** validation/test dataset 缺少训练期 artifact 或其 fingerprint、feature mode、domain policy 不兼容
- **THEN** runtime MUST 在评估前失败或要求显式重新生成训练 artifact
- **AND** runtime MUST NOT 从 validation/test 数据补拟合

#### Scenario: Artifact provenance 完整
- **WHEN** normalization artifact 被写出或复用
- **THEN** metadata MUST 记录模态、fit split、effective sample count、domain policy、feature mode 和稳定 fingerprint
- **AND** fit split MUST 为 train
