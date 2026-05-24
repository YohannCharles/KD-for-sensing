# dataset-runtime-contracts Specification

## Purpose
定义跨数据集 runtime contract：用轻量 dataset descriptor 描述数据集家族与本地产物边界，用 sample index、modality adapter 和 target provider 组合 flat sample，并在训练/评估 runtime metadata 中同时记录 dataset family、storage kind、split、enabled modalities、input profiles 和当前 objective 实际消费的 target schema。
## Requirements
### Requirement: Dataset descriptor 注册契约
系统 MUST 提供 dataset descriptor 或等价机制，用于描述每个数据集家族的 dataset type、默认目录、存储类型、split 语义、支持模态、支持 target schema 和本地产物边界。descriptor 查询 MUST 保持轻量，不得导入 pandas、h5py、torch dataset、模型或训练模块。

#### Scenario: 查询 Multimodal-NF descriptor
- **WHEN** 代码查询 `multimodal_nf` dataset descriptor
- **THEN** 系统 MUST 返回 family 名称、默认根目录 `dataset/MultimodalNF`、存储类型 `hdf5_frame`、默认 split 策略和支持的模态/profile
- **AND** 查询过程 MUST 不读取真实 HDF5 数据、不打开 codebook 文件、不导入训练循环

#### Scenario: 查询旧数据集 descriptor
- **WHEN** 代码查询 DeepSense6G、MMW 或 Raymobtime s008 的 descriptor
- **THEN** 系统 MUST 返回对应存储类型和默认路径
- **AND** 旧 dataset type、配置和公开输出字段 MUST 保持兼容

### Requirement: Sample index 统一契约
系统 MUST 提供 sample index 契约，把 CSV、HDF5、NPZ cache 或 manifest 转换为轻量样本 rows。sample row MUST 至少能表达 `sample_id`、split、数据集家族、scene/city、trajectory、frame、资源引用、target 引用和 metadata。sample index 初始化 MUST 不物化 image、LiDAR、CSI/channel 等大数组。

#### Scenario: HDF5 frame index
- **WHEN** Multimodal-NF index builder 读取 city-level HDF5 metadata
- **THEN** 每个合法 frame MUST 生成一个可追踪 sample row
- **AND** row MUST 包含 city id、trajectory id、frame id 和对应 channel/image/lidar/resource 引用
- **AND** index metadata MUST 记录样本数、city 分布、split 策略和输入文件 fingerprint

#### Scenario: CSV sequence index 兼容
- **WHEN** DeepSense6G 仍使用现有 CSV sequence 样本构建
- **THEN** 系统 MAY 通过适配层暴露 sample index row
- **AND** 该适配 MUST 不改变 `input_beam`、`target_beam`、模态样本字段或 metadata 的既有语义

### Requirement: Modality adapter profile 契约
系统 MUST 支持按模态和 profile 注册 modality adapter。adapter MUST 声明输入字段、所需资源引用、输出 sample key、shape/dtype 语义、cache/normalization 能力和错误信息。dataset 取样 MUST 只调用启用模态对应的 adapter。

#### Scenario: 只加载启用模态
- **WHEN** 用户构建只启用 `gps` 和 `csi` 的 Multimodal-NF dataset
- **THEN** dataset MUST 不读取 image 或 LiDAR HDF5/zip 数据
- **AND** 返回样本 MUST 只包含启用模态字段、目标字段和 metadata

#### Scenario: profile shape 校验
- **WHEN** adapter 读取到的 Multimodal-NF LiDAR 点云不是 `[P, 3]` 或 image 不是 RGB 三通道
- **THEN** 系统 MUST 抛出包含 dataset family、modality、profile、sample_id 和实际 shape 的清晰错误

### Requirement: Target provider 契约
系统 MUST 支持按 objective 或 target schema 注册 target provider。target provider MUST 负责生成主 label、辅助 target、valid mask 和 target metadata，并 MUST 允许 train split 产出的统计或 codebook metadata 复用于 val/test split。

#### Scenario: 近场 beam target
- **WHEN** Multimodal-NF target provider 读取 Top-5 三维 beam codebook 标签
- **THEN** provider MUST 返回 flattened `target_beam`
- **AND** provider MUST 保留 Top-5 triplet、beam power、codebook shape 和 flatten 规则 metadata

#### Scenario: Artifact 复用
- **WHEN** train dataset 已解析 codebook metadata 或 normalizer artifact
- **THEN** data factory MUST 能将需要复用的 artifact 传给 val/test dataset
- **AND** 复用过程 MUST 不要求重新扫描全量大数据文件

### Requirement: RuntimeDataset flat sample 契约
系统 MUST 提供薄 runtime dataset 或等价组合方式，通过 sample index、modality adapters 和 target provider 构建 flat dict sample。flat sample keys MUST 与中心化模态契约和 prediction objective 契约一致。

#### Scenario: Flat sample 输出
- **WHEN** 用户从任一支持数据集取样
- **THEN** 返回值 MUST 是 DataLoader 可默认 collate 的 flat dict
- **AND** 输入模态字段 MUST 使用中心化模态契约定义的 sample key
- **AND** target 字段 MUST 使用当前 objective 或 target schema 定义的字段名

#### Scenario: Runtime metadata
- **WHEN** 训练或评估构建 dataloaders
- **THEN** run metadata MUST 记录 dataset type、descriptor family、storage kind、split metadata、enabled modalities、input profiles 和 target schema

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

