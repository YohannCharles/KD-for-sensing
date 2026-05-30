## MODIFIED Requirements

### Requirement: descriptor 驱动 dataset 构建
数据构建流程 MUST 根据当前保留 dataset descriptor 决定 split 解析、默认路径、storage kind、enabled modalities、input profiles 和 target schema。非 CSV 数据集 MUST 不被强制套用 DeepSense6G 的 train/test CSV 规则。已退役的 `multimodal_nf` descriptor、HDF5 index 和 cache index MUST 不再作为支持构建路径。

#### Scenario: Multimodal-NF dataset 构建失败
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** data factory MUST 拒绝该 dataset type
- **AND** 系统 MUST 不解析 `dataset/MultimodalNF`、HDF5 index 或 cache index

#### Scenario: 构建 DeepSense6G CSV dataset
- **WHEN** 用户配置 `data.dataset.type: deepsense6g`
- **THEN** data factory MUST 继续使用 DeepSense6G scene 和 split CSV 规则
- **AND** 现有 `train_csv_name`、`val_csv_name`、`test_csv_name` 覆盖行为 MUST 保持兼容

### Requirement: enabled modalities 与 profile 一起传递
数据构建流程 MUST 从实验任务、fusion 模态和当前保留 dataset descriptor 推导启用模态，并将标准化后的 input profiles 传递给 dataset、batch 准备和 run metadata。系统 MUST 不再解析或传递 Multimodal-NF 专属 profiles。

#### Scenario: 保留 fusion profile 传递
- **WHEN** 用户运行当前保留 fusion 配置并启用多个模态
- **THEN** data factory MUST 设置 `enabled_modalities`
- **AND** data factory MUST 传递每个保留模态的 resolved profile
- **AND** run metadata MUST 记录实际启用模态和 profile

#### Scenario: Multimodal-NF profile 拒绝
- **WHEN** 用户为退役 Multimodal-NF 配置 `image`、`lidar`、`gps` 或 `csi` profile
- **THEN** 系统 MUST 拒绝该 dataset type 或 profile
- **AND** 错误信息 MUST 指出 Multimodal-NF 已退役

### Requirement: HDF5/cache-backed 懒加载
HDF5 或 cache-backed dataset 如果属于当前保留数据集，MUST 在初始化时只读取 index、shape 和 metadata，不得物化全量 image、LiDAR、CSI/channel 大数组。未启用模态 MUST 完全跳过对应数据读取。Multimodal-NF HDF5/cache-backed lazy loading 不再作为支持路径。

#### Scenario: 保留 dataset 初始化不物化大数组
- **WHEN** 用户构建当前保留的 HDF5/cache-backed train dataset
- **THEN** dataset 初始化 MUST 不把全量大数组读入内存
- **AND** dataset MUST 只保留轻量 index、文件路径、key 和必要 metadata

#### Scenario: Multimodal-NF lazy loading 删除
- **WHEN** 用户运行 Multimodal-NF CSI-only、image-only 或 fusion 配置
- **THEN** dataset 构建 MUST 失败
- **AND** 系统 MUST 不进入 Multimodal-NF image、LiDAR 或 CSI lazy loading 分支

### Requirement: split metadata 和 artifact 复用
数据构建流程 MUST 记录并复用当前保留数据集的 split metadata 和必要 normalizer artifact。train split 拟合出的 artifact MUST 能传递给 val/test split，而不要求重新扫描全量数据。系统 MUST 不再复用 Multimodal-NF split metadata 或 codebook metadata。

#### Scenario: 保留 artifact 复用
- **WHEN** train dataset 已解析当前保留数据集所需的 normalizer 或 metadata
- **THEN** val/test dataset MUST 使用同一 metadata 或 artifact
- **AND** 如果 val/test metadata 与 train 不一致，系统 MUST 抛出清晰错误

#### Scenario: Multimodal-NF metadata 删除
- **WHEN** 训练或评估构建 dataloaders
- **THEN** run metadata MUST 不要求包含 Multimodal-NF split protocol、city 列表、input profiles、target schema 或 codebook metadata
