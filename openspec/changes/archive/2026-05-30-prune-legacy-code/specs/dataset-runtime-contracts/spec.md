## MODIFIED Requirements

### Requirement: Dataset descriptor 注册契约
系统 MUST 提供 dataset descriptor 或等价机制，用于描述当前保留数据集家族的 dataset type、默认目录、存储类型、split 语义、支持模态、支持 target schema 和本地产物边界。descriptor 查询 MUST 保持轻量，不得导入 pandas、h5py、torch dataset、模型或训练模块。已退役的 `multimodal_nf` descriptor MUST 不再注册。

#### Scenario: 查询退役 Multimodal-NF descriptor
- **WHEN** 代码查询 `multimodal_nf` dataset descriptor
- **THEN** 系统 MUST 报告该 dataset type 不存在或已退役
- **AND** 查询过程 MUST 不读取真实 HDF5 数据、不打开 codebook 文件、不导入训练循环

#### Scenario: 查询旧数据集 descriptor
- **WHEN** 代码查询 DeepSense6G、MMW 或 Raymobtime s008 的 descriptor
- **THEN** 系统 MUST 返回对应存储类型和默认路径
- **AND** 这些保留 dataset type、配置和公开输出字段 MUST 保持兼容

### Requirement: Runtime metadata 区分 dataset family 与 target schema
Dataset runtime metadata MUST 同时记录当前保留 dataset family 信息和当前 objective target schema。dataset family MUST 表达数据来源、storage kind、split 和 profiles；target schema MUST 表达当前 run 实际训练或评估的主 target 和辅助 target。系统 MUST 不再写出或要求 Multimodal-NF runtime metadata。

#### Scenario: 保留数据集 metadata 双层记录
- **WHEN** 训练或评估构建当前保留数据集 dataloaders
- **THEN** runtime metadata MUST 记录 dataset type、storage kind、split strategy、enabled modalities 和 input profiles
- **AND** runtime metadata MUST 记录当前 objective 对应的 target schema
- **AND** 二者 MUST 不互相覆盖

#### Scenario: Raymobtime 与其它保留数据集语义隔离
- **WHEN** 系统写出 Raymobtime s008 和其它保留数据集 run metadata
- **THEN** Raymobtime current snapshot beam selection MUST 使用 Raymobtime task semantics
- **AND** 其它保留数据集 MUST 使用各自 target schema
- **AND** 系统 MUST 不写出 Multimodal-NF near-field target schema

### Requirement: Sample index 统一契约
系统 MUST 提供 sample index 契约，把当前保留数据集使用的 CSV、NPZ cache 或 manifest 转换为轻量样本 rows。sample row MUST 至少能表达 `sample_id`、split、数据集家族、scene/condition、trajectory、frame、资源引用、target 引用和 metadata。sample index 初始化 MUST 不物化 image、LiDAR、CSI/channel 等大数组。系统 MUST 不再要求支持 Multimodal-NF HDF5 frame index。

#### Scenario: CSV sequence index 兼容
- **WHEN** DeepSense6G 仍使用现有 CSV sequence 样本构建
- **THEN** 系统 MUST 允许通过适配层暴露 sample index row
- **AND** 该适配 MUST 不改变 `input_beam`、`target_beam`、模态样本字段或 metadata 的既有语义

#### Scenario: Multimodal-NF HDF5 frame index 不再支持
- **WHEN** 用户查找或调用 Multimodal-NF HDF5 frame index builder
- **THEN** 系统 MUST 报告该 builder 不存在或 dataset type 已退役
- **AND** 系统 MUST 不读取 Multimodal-NF HDF5 文件

### Requirement: Modality adapter profile 契约
系统 MUST 支持当前保留数据集按模态和 profile 注册 modality adapter。adapter MUST 声明输入字段、所需资源引用、输出 sample key、shape/dtype 语义、cache/normalization 能力和错误信息。dataset 取样 MUST 只调用启用模态对应的 adapter。系统 MUST 不再提供 Multimodal-NF 专属 adapter/profile。

#### Scenario: 只加载启用模态
- **WHEN** 用户构建当前保留的多模态 dataset 且只启用部分模态
- **THEN** dataset MUST 不读取未启用模态的数据源
- **AND** 返回样本 MUST 只包含启用模态字段、目标字段和 metadata

#### Scenario: Multimodal-NF adapter 删除
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** 系统 MUST 不再调用 Multimodal-NF image、LiDAR、GPS 或 CSI adapter
- **AND** dataset 构建 MUST 失败

### Requirement: Target provider 契约
系统 MUST 支持当前保留 objectives 或 target schema 的 target provider。target provider MUST 负责生成主 label、辅助 target、valid mask 和 target metadata，并 MUST 允许 train split 产出的统计或 metadata 复用于 val/test split。系统 MUST 不再提供 Multimodal-NF 近场三维 codebook target provider。

#### Scenario: Artifact 复用
- **WHEN** train dataset 已解析当前保留 objective 所需的 normalizer artifact 或 metadata
- **THEN** data factory MUST 能将需要复用的 artifact 传给 val/test dataset
- **AND** 复用过程 MUST 不要求重新扫描全量大数据文件

#### Scenario: 近场 beam provider 删除
- **WHEN** batch 或配置请求 Multimodal-NF Top-5 三维 beam target
- **THEN** 系统 MUST 不再提供该 target provider
- **AND** 错误信息 MUST 指出 Multimodal-NF 已退役
