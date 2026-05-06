# modality-contracts Specification

## Purpose
TBD - created by archiving change clarify-architecture-boundaries. Update Purpose after archive.
## Requirements
### Requirement: 中心化模态契约
项目 MUST 提供单一来源的模态契约，用于描述所有受支持模态的规范名称、固定顺序、dataset flag、样本字段、fusion 输入字段、默认 dataset/model 字段，以及是否支持 cache 或归一化 artifact。

#### Scenario: 枚举受支持模态
- **WHEN** 开发者查询模态契约
- **THEN** 系统 MUST 返回固定顺序的 `image`、`radar`、`gps`、`lidar` 和 `mmwave`
- **AND** 该顺序 MUST 被 canonical config、fusion 模态解析、dataset 构建和诊断配置复用

#### Scenario: 查询单个模态元数据
- **WHEN** 开发者查询 `radar` 模态契约
- **THEN** 系统 MUST 返回 radar 对应的样本字段 `radar_ra` 和 `radar_da`
- **AND** 系统 MUST 返回 fusion 输入字段 `radar_batch`

### Requirement: 模态列表标准化
系统 MUST 通过模态契约标准化用户配置中的模态列表。标准化 MUST 拒绝未知模态、空列表和重复模态，并 MUST 按固定模态顺序返回结果。

#### Scenario: 标准化乱序 fusion 模态
- **WHEN** 用户配置 fusion `modalities: ["lidar", "image", "gps"]`
- **THEN** 系统 MUST 将有效模态标准化为 `["image", "gps", "lidar"]`
- **AND** dataset、batch 准备和模型构建 MUST 使用同一个标准化结果

#### Scenario: 拒绝未知模态
- **WHEN** 用户配置 `modalities: ["image", "thermal"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知模态和可用模态列表

#### Scenario: 拒绝重复模态
- **WHEN** 用户配置 `modalities: ["gps", "gps"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出模态列表包含重复项

### Requirement: 模态契约驱动 dataset flag
数据构建流程 MUST 通过模态契约生成 dataset flag 和相关默认字段，避免在多个模块中手写 `use_gps`、`use_lidar`、`use_mmwave` 等分支。

#### Scenario: GPS 与 mmWave fusion 设置 dataset flag
- **WHEN** fusion 启用模态为 `["gps", "mmwave"]`
- **THEN** 数据构建流程 MUST 设置 `use_gps: true` 和 `use_mmwave: true`
- **AND** 数据构建流程 MUST 不设置与未启用 LiDAR 相关的启用 flag

#### Scenario: 单模态 image 不启用可选模态 flag
- **WHEN** `experiment.task: image`
- **THEN** 数据构建流程 MUST 只启用 image 所需字段
- **AND** GPS、LiDAR 和 mmWave 的 dataset flag MUST 保持关闭或缺省关闭

### Requirement: 模态契约驱动 batch 输入
训练、验证、评估和诊断路径 MUST 使用模态契约确定 batch 字段到模型输入参数的映射。新增或调整模态输入键时，系统 MUST 不要求在训练、验证和评估循环中复制分支逻辑。

#### Scenario: fusion batch 输入映射
- **WHEN** fusion 启用模态为 `["radar", "mmwave"]`
- **THEN** batch 准备流程 MUST 从样本字段构建 `radar_batch` 和 `mmwave_batch`
- **AND** 模型调用 MUST 不传入未启用 image、GPS 或 LiDAR 的输入张量

#### Scenario: 单模态 batch 输入映射
- **WHEN** `experiment.task: lidar`
- **THEN** batch 准备流程 MUST 构建 LiDAR 模型所需输入
- **AND** 训练和评估循环 MUST 通过统一映射调用模型

### Requirement: 模态契约文档化
项目文档 MUST 说明模态契约的职责，以及新增模态时必须补充的 dataset 字段、batch 字段、model 输入、cache/normalization 能力和测试。

#### Scenario: 按文档新增模态
- **WHEN** 开发者按照扩展文档新增一个实验性模态
- **THEN** 文档 MUST 指引开发者先新增模态契约
- **AND** 文档 MUST 指出后续需要补充 dataset 读取、模型注册、batch 准备和诊断显示逻辑

