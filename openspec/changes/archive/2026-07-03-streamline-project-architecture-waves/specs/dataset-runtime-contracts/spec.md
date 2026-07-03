## ADDED Requirements

### Requirement: Dataset families use composition over inheritance growth
项目 MUST 将 DeepSense6G/MMW dataset 的长期扩展边界收敛为组合式 dataset family adapter。新增 dataset family、模态 reader、target、metadata 或物理监督时，主要实现 MUST 位于 family adapter、resource reader、target provider、sample assembly 或 contract helper 中，而不是继续扩大 `DeepSense6GDataset` / `MMWDataset` 的 `__init__` 或 `__getitem__` 主体。

#### Scenario: 新增 dataset family 行为
- **WHEN** 开发者为 MMW 或其它 dataset family 增加 CSV 补列、layout、geometry、availability、radio/path semantic、physical label 或 physics supervision 行为
- **THEN** 主要实现 MUST 位于 dataset-family adapter 或对应窄 helper
- **AND** registry owner class MAY 保持现有 `DATASETS.build()` public 行为，但 MUST 委托 adapter，而不是继续堆叠继承逻辑

#### Scenario: 新增模态 reader
- **WHEN** 开发者新增或修改 image、radar、GPS、LiDAR、mmWave、CSI 或未来 current modality 的资源读取方式
- **THEN** 主要实现 MUST 位于 modality reader 或 transform owner
- **AND** dataset 主体 MUST 只连接 reader 输出到统一 sample contract

### Requirement: Dataset sample assembly remains schema-compatible
Dataset 重构 MUST 保持 batch sample key、target tensor、metadata、domain metadata、soft label、auxiliary target、cache key、portion sampling 和 enabled modality semantics 兼容。任何字段新增 MUST 是 opt-in 或由现有 contract 明确允许，普通 baseline MUST 不因新增 metadata 变为必需输入。

#### Scenario: Dataset adapter 输出兼容
- **WHEN** 重构后的 DeepSense6G 或 MMW dataset 返回样本
- **THEN** 现有训练、评估、diagnostics、difficulty pipeline 和 batch preparation 所依赖的 sample keys MUST 继续存在
- **AND** 新增 metadata MUST 不改变 `target_beam`、`input_beam`、`beam_power`、split metadata 或 sample id 的既有语义

#### Scenario: Cache 和 scaler 行为兼容
- **WHEN** dataset adapter 化移动 image/LiDAR/sample cache、GPS/mmWave/LiDAR/CSI normalizer 或 target scaler 逻辑
- **THEN** artifact save/load 格式和 normalization runtime metadata MUST 保持兼容
- **AND** focused tests MUST 覆盖至少一个 synthetic 或 fixture 路径，不得依赖真实 `dataset/`

### Requirement: Dataset hotspot growth is guarded
项目健康护栏 MUST 防止新的 dataset contract 规则继续堆入 dataset 巨型 class。新增超长 `__init__`、`__getitem__`、family-specific branch 或 modality-specific branch 时，架构边界测试或 focused tests MUST 要求拆分到 adapter/helper 或更新 inventory 暂缓理由。

#### Scenario: 新规则进入 helper
- **WHEN** 新增 GPS feature mode、beam target source、label-space validation、cache path、position/occlusion/physics target 或 MMW semantic rule
- **THEN** 主要实现 MUST 位于对应 contract/helper/adapter
- **AND** dataset class 主体只能承担薄 orchestration 和 public registry owner 职责

