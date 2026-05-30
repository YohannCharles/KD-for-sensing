## REMOVED Requirements

### Requirement: Multimodal-NF 吞吐 profile
**Reason**: Multimodal-NF dataset 和训练配置退役，相关 image/LiDAR/GPS/CSI/fusion profiling 不再维护。
**Migration**: 使用当前保留数据集的吞吐 profiling。

#### Scenario: Multimodal-NF profile 删除
- **WHEN** 用户请求 Multimodal-NF 吞吐 profile
- **THEN** 系统 MUST 不再提供该 profile 路径
- **AND** profile 输出 MUST 不要求包含 Multimodal-NF 模态级 getitem 字段

### Requirement: Multimodal-NF 吞吐配置推荐
**Reason**: Multimodal-NF image/LiDAR/fusion 配置和 cache 策略退役。
**Migration**: 并行训练和吞吐推荐只覆盖当前保留数据集。

#### Scenario: Multimodal-NF 推荐删除
- **WHEN** 用户请求 Multimodal-NF image+LiDAR+GPS fusion 的吞吐推荐
- **THEN** 系统 MUST 报告该 dataset/config 已退役

### Requirement: Multimodal-NF 吞吐回归验证
**Reason**: Multimodal-NF fixture、原始 HDF5 与派生缓存等价验证随数据集退役。
**Migration**: 使用当前保留数据集的吞吐回归测试。

#### Scenario: Multimodal-NF 吞吐测试删除
- **WHEN** 开发者运行 focused throughput tests
- **THEN** 测试 MUST 不再要求 Multimodal-NF fixture 或派生缓存等价性

### Requirement: Multimodal-NF cache IO profiling
**Reason**: Multimodal-NF image/LiDAR 派生 cache 退役。
**Migration**: 当前保留 cache 的 IO profiling 由对应 specs 约束。

#### Scenario: Multimodal-NF cache profile 删除
- **WHEN** profile 读取 runtime metadata
- **THEN** 系统 MUST 不要求解析 Multimodal-NF cache validation、open 或 read 耗时字段

### Requirement: Multimodal-NF train 子采样局部性控制
**Reason**: 该局部性策略服务于退役的大 cache 训练场景。
**Migration**: 通用 train 子采样能力如仍保留，由非 Multimodal-NF 要求约束。

#### Scenario: Multimodal-NF 局部性策略删除
- **WHEN** 用户配置 train 子采样
- **THEN** 系统 MUST 不提供 Multimodal-NF image/LiDAR cache locality 专属策略

### Requirement: Multimodal-NF 并行训练 IO 推荐
**Reason**: Multimodal-NF 并行训练 IO 推荐随数据集退役。
**Migration**: 当前保留数据集的并行训练推荐继续有效。

#### Scenario: Multimodal-NF 并行推荐删除
- **WHEN** 用户请求多个 Multimodal-NF 后台训练并行运行建议
- **THEN** 系统 MUST 报告该工作流已退役

### Requirement: Multimodal-NF cache migration 诊断
**Reason**: Multimodal-NF cache sidecar migration 退役。
**Migration**: 不迁移历史 sidecar。

#### Scenario: cache migration 诊断删除
- **WHEN** 用户运行吞吐 profile 或并行训练推荐
- **THEN** 系统 MUST 不要求输出 Multimodal-NF cache migration pending 数量或维护建议
