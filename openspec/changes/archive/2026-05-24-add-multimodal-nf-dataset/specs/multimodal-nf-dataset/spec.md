## ADDED Requirements

### Requirement: Multimodal-NF 本地数据布局与审计
系统 MUST 支持 `multimodal_nf` dataset type，并将默认数据集家族目录解析为 `dataset/MultimodalNF`。系统 MUST 提供配置驱动审计流程，用于检查 Multimodal-NF HDF5、image/lidar 压缩包或 HDF5、codebook 文件、city 列表、HDF5 keys、shape、dtype 和 split 可用性。

#### Scenario: 默认目录解析
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf` 且未设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/MultimodalNF` 作为默认数据根目录
- **AND** 该路径 MUST 通过项目根路径解析工具解析

#### Scenario: 审计官方数据文件
- **WHEN** 用户运行 Multimodal-NF 审计预处理
- **THEN** 系统 MUST 检查 channel HDF5、image 数据、LiDAR 数据和 `upa64x64_NF_codebook*.pkl` 是否存在
- **AND** 审计输出 MUST 记录每个 HDF5 文件的 keys、shape、dtype、city id、样本数和缺失项
- **AND** 审计输出 MUST 写入 `outputs/`、`dataset/MultimodalNF/cache` 或用户配置的 ignored 目录

#### Scenario: 不自动迁移真实数据
- **WHEN** 本地存在 Hugging Face 下载目录、外部数据目录或旧目录结构
- **THEN** 系统 MUST 不自动移动、复制或删除真实数据文件
- **AND** 用户 MUST 能通过显式 `data_root`、`channel_root`、`image_root`、`lidar_root` 或 `codebook_path` 继续使用该路径

### Requirement: Multimodal-NF HDF5 index 构建
系统 MUST 从 Multimodal-NF HDF5 和 metadata 构建 frame-wise sample index。每个样本 MUST 对应单个 UAV trajectory frame，并 MUST 可追踪到 city、trajectory 和 frame。

#### Scenario: 构建 frame-wise index
- **WHEN** index builder 读取 `City_###` 数据
- **THEN** 每个 frame MUST 生成一个稳定 `sample_id`
- **AND** `sample_id` MUST 包含或可反查 city id、trajectory id 和 frame id
- **AND** index row MUST 包含 channel、image、LiDAR、position、beam target 和辅助标签引用

#### Scenario: city-level split
- **WHEN** 用户使用默认 split 策略
- **THEN** 系统 MUST 按 city 或配置定义的 city 集合划分 train/validation/test
- **AND** 同一 city 的样本 MUST 不同时出现在多个 split 中，除非配置显式选择 frame-level debug split
- **AND** split metadata MUST 记录 city 列表、样本数、beam label 分布、LoS/NLoS 分布和 NF/FF 分布

### Requirement: Multimodal-NF dataset sample 契约
`multimodal_nf` dataset MUST 返回 flat dict sample，并根据启用模态和 profile 懒加载输入。dataset MUST 不读取未启用模态对应的大数组或 HDF5 dataset。

#### Scenario: GPS+CSI 取样
- **WHEN** 用户启用 `gps` profile `uav_xyz_snapshot` 和 `csi` profile `xl_mimo_nf`
- **THEN** 样本 MUST 包含 `gps` 和 `csi`
- **AND** `gps` MUST 具有 `[1, 3]` 当前 UAV 位置语义
- **AND** `csi` MUST 具有 `[1, 4096, K, 2]` 或配置选择后的等价 near-field XL-MIMO channel 语义
- **AND** 样本 MUST 不包含 `image` 或 `lidar`

#### Scenario: Image+LiDAR 取样
- **WHEN** 用户启用 `image` 和 `lidar` profile `point_cloud_xyz_10000`
- **THEN** `image` MUST 返回 `[1, 3, H, W]` RGB tensor
- **AND** `lidar` MUST 返回 `[1, P, 3]` 点云 tensor，默认 P 为 10000
- **AND** 如果原始 image 为 `[H, W, 3]`，dataset 或 adapter MUST 转换为 channel-first RGB tensor

#### Scenario: Metadata 字段
- **WHEN** 用户设置 `return_metadata: true`
- **THEN** 样本 metadata MUST 至少包含 `sample_id`、`dataset_type`、`city_id`、`trajectory_id`、`frame_id`、`split`、启用 profile 和源文件路径或引用

#### Scenario: split-specific 验证抽样
- **WHEN** 用户为 Multimodal-NF dataset 配置 `eval_portion` 或 split-specific portion
- **THEN** 系统 MUST 只对对应 validation/test split 应用该抽样
- **AND** train split MUST 默认保持 `portion` 或 `train_portion` 指定的样本范围
- **AND** run metadata MUST 记录实际应用的 selected portion

### Requirement: Multimodal-NF 近场 beam target 契约
Multimodal-NF dataset MUST 支持三维 codebook Top-5 beam target。主训练标签 MUST 以 `target_beam` 暴露为 flattened class，同时保留 triplet 与 power 信息用于指标、诊断和后续结构化方法。

#### Scenario: Top-5 target 输出
- **WHEN** 样本包含 `BeamIdx` 和 `BeamPower`
- **THEN** dataset MUST 返回 `target_beam` 为 Top-1 triplet flatten 后的一维当前标签
- **AND** dataset MUST 返回 `beam_triplet_topk`，形状为 `[5, 3]`
- **AND** dataset MUST 返回 `beam_power_topk`，形状为 `[5]`
- **AND** dataset metadata 或 dataset 属性 MUST 记录 codebook shape 和 flatten 规则

#### Scenario: codebook 文件缺失
- **WHEN** 用户启用 near-field beam target 但 codebook 文件或 codebook shape 不可解析
- **THEN** 系统 MUST 拒绝构建 dataset
- **AND** 错误信息 MUST 指出缺失的 codebook 配置和可用路径

### Requirement: Multimodal-NF 辅助标签契约
Multimodal-NF dataset MUST 暴露 LoS/NLoS、NF/FF、trajectory mode 或等价辅助标签，用于分析、过滤和可选辅助任务。

#### Scenario: 辅助标签输出
- **WHEN** 原始 HDF5 包含 `Has_LoS`、`Is_NF`、`Traj_Is_NLoS` 或 `Mode_Idx`
- **THEN** dataset MUST 将可用标签以 `los_label`、`nf_label`、`traj_nlos_label` 和 `mode_idx` 或等价规范字段输出
- **AND** 缺失某个辅助标签时系统 MUST 在 metadata 中记录不可用状态，而不是静默填入错误值

### Requirement: Multimodal-NF 配置和 smoke workflow
系统 MUST 提供 Multimodal-NF 预处理、dataset smoke、单模态 near-field beam selection 和 fusion 配置样例。smoke workflow MUST 可在无真实全量数据的情况下通过小型 fixture 验证核心契约。

#### Scenario: 预处理配置
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/multimodal_nf_audit.yaml`
- **THEN** 命令 MUST 能输出审计报告或清晰的数据缺失错误
- **AND** 命令 MUST 不要求提交任何真实数据或生成 cache 到源码变更

#### Scenario: smoke 测试
- **WHEN** 开发者运行 Multimodal-NF focused tests
- **THEN** 测试 MUST 使用小型 fixture 验证 HDF5 index、dataset fields、target flatten、Top-5 metadata、profile shape 和 data factory 构建
- **AND** 测试命令 MUST 使用 `conda run -n kd_mm_beam pytest ... -q`
