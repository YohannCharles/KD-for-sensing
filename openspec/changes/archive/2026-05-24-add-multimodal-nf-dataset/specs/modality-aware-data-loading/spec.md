## ADDED Requirements

### Requirement: descriptor 驱动 dataset 构建
数据构建流程 MUST 根据 dataset descriptor 决定 split 解析、默认路径、storage kind、enabled modalities、input profiles 和 target schema。非 CSV 数据集 MUST 不被强制套用 DeepSense6G 的 train/test CSV 规则。

#### Scenario: 构建 Multimodal-NF HDF5 dataset
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** data factory MUST 使用 Multimodal-NF descriptor 解析默认根目录、HDF5 index 或 cache index
- **AND** data factory MUST 不要求 `train_csv_name`、`test_csv_name` 或 `csv_name`
- **AND** dataset split MUST 来自 Multimodal-NF index metadata、city split 配置或显式 split 文件

#### Scenario: 构建 DeepSense6G CSV dataset
- **WHEN** 用户配置 `data.dataset.type: deepsense6g`
- **THEN** data factory MUST 继续使用 DeepSense6G scene 和 split CSV 规则
- **AND** 现有 `train_csv_name`、`val_csv_name`、`test_csv_name` 覆盖行为 MUST 保持兼容

### Requirement: enabled modalities 与 profile 一起传递
数据构建流程 MUST 从实验任务、fusion 模态和 dataset descriptor 推导启用模态，并将标准化后的 input profiles 传递给 dataset、batch 准备和 run metadata。

#### Scenario: Multimodal-NF fusion profile 传递
- **WHEN** 用户运行 Multimodal-NF fusion 配置，启用 `["image", "lidar", "gps", "csi"]`
- **THEN** data factory MUST 设置 `enabled_modalities`
- **AND** data factory MUST 传递每个模态的 resolved profile
- **AND** run metadata MUST 记录 `image: rgb_imagenet`、`lidar: point_cloud_xyz_10000`、`gps: uav_xyz_snapshot` 和 `csi: xl_mimo_nf`

#### Scenario: profile 与任务冲突
- **WHEN** 用户启用 `experiment.task: csi` 但为非 Multimodal-NF 数据集配置 `csi_profile: xl_mimo_nf`
- **THEN** 系统 MUST 根据 descriptor 拒绝不受支持的 profile
- **AND** 错误信息 MUST 指出 dataset type、modality 和 profile 冲突

### Requirement: HDF5/cache-backed 懒加载
HDF5 或 cache-backed dataset MUST 在初始化时只读取 index、shape 和 metadata，不得物化全量 image、LiDAR、CSI/channel 大数组。未启用模态 MUST 完全跳过对应 HDF5 dataset 或 zip 数据读取。

#### Scenario: 初始化不物化大数组
- **WHEN** 用户构建 Multimodal-NF train dataset
- **THEN** dataset 初始化 MUST 不把全量 `H`、`image` 或 `points` 数据读入内存
- **AND** dataset MUST 只保留轻量 index、文件路径、HDF5 key 和必要 metadata

#### Scenario: 未启用 image 不打开 image 数据
- **WHEN** 用户运行 Multimodal-NF CSI-only 配置
- **THEN** dataset 取样 MUST 不读取 image HDF5 或 image zip
- **AND** image 文件缺失 MUST 不阻止 CSI-only 任务运行，除非审计或配置显式要求完整多模态数据

### Requirement: split metadata 和 artifact 复用
数据构建流程 MUST 记录并复用 Multimodal-NF split metadata、codebook metadata 和必要 normalizer artifact。train split 拟合出的 artifact MUST 能传递给 val/test split，而不要求重新扫描全量数据。

#### Scenario: codebook metadata 复用
- **WHEN** train dataset 已解析 codebook shape 和 flatten 规则
- **THEN** val/test dataset MUST 使用同一 metadata
- **AND** 如果 val/test metadata 与 train 不一致，系统 MUST 抛出清晰错误

#### Scenario: split metadata 写入 run metadata
- **WHEN** 训练或评估构建 Multimodal-NF dataloaders
- **THEN** run metadata MUST 包含 split protocol、city 列表、样本数、enabled modalities、input profiles、target schema 和 codebook metadata
