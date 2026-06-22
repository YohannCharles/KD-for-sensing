## ADDED Requirements

### Requirement: Dataset descriptor 可复用模态合约
Dataset descriptor 或等价查询机制 MUST 优先复用中心化 `modalities.py` 中的 modality/profile 合约，避免维护第二套静态 profile/dataclass 镜像。保留 descriptor API 时，它 MUST 保持轻量，不导入 pandas、torch dataset、模型或训练模块。

#### Scenario: 查询 input profile
- **WHEN** config validation 或 data factory 解析 enabled modalities 的 input profiles
- **THEN** profile 名称、sample key、fusion input key 和 shape/metadata MUST 来自中心化模态合约或与其等价的单一数据源
- **AND** 删除重复 descriptor dataclass MUST 不改变 current config 的 resolved profiles

#### Scenario: 保留轻量 descriptor 行为
- **WHEN** 代码查询 DeepSense6G 或 MMW dataset descriptor
- **THEN** 查询 MUST 返回 dataset family、storage kind、默认 root、split semantics 和 artifact boundary
- **AND** 查询 MUST 不导入重依赖或读取真实数据

### Requirement: Runtime row 类型不得独立成框架
若 target-shot split、metadata 或 dataset helper 只需要轻量 row 数据，系统 MUST 使用 `Mapping[str, Any]`、flat dict sample 或局部 dataclass。独立 runtime framework 文件只有在被 current workflow 多处消费时才可保留。

#### Scenario: target-shot split 消费 Mapping
- **WHEN** target-shot split 读取样本 row 或 split artifact metadata
- **THEN** 它 MUST 能消费 `Mapping[str, Any]` 或 flat dict sample
- **AND** 它 MUST 不要求独立 `SampleRow` framework 文件存在

#### Scenario: 删除未消费 row framework
- **WHEN** 独立 row/dataclass 文件没有 registry、CLI、current docs/OpenSpec 或多个 runtime 调用方
- **THEN** 本 change MAY 删除该文件或把类型迁入实际 owner
- **AND** dataset runtime metadata 和 target-shot split tests MUST 继续通过
