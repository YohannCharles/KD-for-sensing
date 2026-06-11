## MODIFIED Requirements

### Requirement: Dataset descriptor 注册契约
系统 MUST 提供 dataset descriptor 或等价机制，用于描述当前保留数据集家族的 dataset type、默认目录、存储类型、split 语义、支持模态、支持 target schema 和本地产物边界。descriptor 查询 MUST 保持轻量，不得导入 pandas、h5py、torch dataset、模型或训练模块。已退役的 `multimodal_nf` 和 `raymobtime_s008` descriptor MUST 不再注册。

#### Scenario: 查询退役 Multimodal-NF descriptor
- **WHEN** 代码查询 `multimodal_nf` dataset descriptor
- **THEN** 系统 MUST 报告该 dataset type 不存在或已退役
- **AND** 查询过程 MUST 不读取真实 HDF5 数据、不打开 codebook 文件、不导入训练循环

#### Scenario: 查询退役 Raymobtime descriptor
- **WHEN** 代码查询 `raymobtime_s008` dataset descriptor
- **THEN** 系统 MUST 报告该 dataset type 不存在或已退役
- **AND** 查询过程 MUST 不读取 `dataset/Raymobtime/s008`、不导入 Raymobtime dataset、不导入模型或训练循环

#### Scenario: 查询保留数据集 descriptor
- **WHEN** 代码查询 DeepSense6G 或 MMW 的 descriptor
- **THEN** 系统 MUST 返回对应存储类型和默认路径
- **AND** 这些保留 dataset type、配置和公开输出字段 MUST 保持兼容

### Requirement: Runtime metadata 区分 dataset family 与 target schema
Dataset runtime metadata MUST 同时记录当前保留 dataset family 信息和当前 objective target schema。dataset family MUST 表达数据来源、storage kind、split 和 profiles；target schema MUST 表达当前 run 实际训练或评估的主 target 和辅助 target。系统 MUST 不再写出或要求 Multimodal-NF runtime metadata，也 MUST 不再写出或要求 Raymobtime s008 runtime metadata。

#### Scenario: 保留数据集 metadata 双层记录
- **WHEN** 训练或评估构建当前保留数据集 dataloaders
- **THEN** runtime metadata MUST 记录 dataset type、storage kind、split strategy、enabled modalities 和 input profiles
- **AND** runtime metadata MUST 记录当前 objective 对应的 target schema
- **AND** 二者 MUST 不互相覆盖

#### Scenario: 退役 Raymobtime metadata 不再写出
- **WHEN** 用户加载旧 Raymobtime s008 配置或旧 Raymobtime checkpoint metadata
- **THEN** 当前训练/评估 runtime MUST 不写出新的 `raymobtime_s008_current_snapshot` metadata
- **AND** 系统 MUST 报告 Raymobtime s008 已退役或要求用户使用当前保留 workflow

#### Scenario: Multimodal-NF metadata 不再写出
- **WHEN** 用户加载 Multimodal-NF 配置
- **THEN** 系统 MUST 不写出 Multimodal-NF near-field target schema
- **AND** 系统 MUST 报告该 dataset type 已退役
