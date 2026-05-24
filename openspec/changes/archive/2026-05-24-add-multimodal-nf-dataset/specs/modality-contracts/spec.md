## ADDED Requirements

### Requirement: 模态 profile 契约
中心化模态契约 MUST 支持 dataset-specific input profile，用于在不新增模态名称的情况下表达同一模态在不同数据集中的输入语义、shape、默认字段和 batch 准备规则。profile 标准化 MUST 拒绝未知 profile，并 MUST 保持未配置旧 profile 时的既有行为。

#### Scenario: 查询 GPS UAV 位置 profile
- **WHEN** 开发者查询 `gps` 模态的 `uav_xyz_snapshot` profile
- **THEN** 系统 MUST 返回 sample key `gps`
- **AND** 系统 MUST 返回 fusion input key `gps_batch`
- **AND** 系统 MUST 返回输入语义为当前 UAV 3D 位置 `[T, 3]`，其中 Multimodal-NF 默认 T 为 1
- **AND** DeepSense6G 旧 GPS relative-polar 默认行为 MUST 不受影响

#### Scenario: 查询 CSI 近场 XL-MIMO profile
- **WHEN** 开发者查询 `csi` 模态的 `xl_mimo_nf` profile
- **THEN** 系统 MUST 返回 sample key `csi`
- **AND** 系统 MUST 返回 fusion input key `csi_batch`
- **AND** 系统 MUST 返回输入语义为近场 XL-MIMO channel tensor `[T, M, K, 2]`
- **AND** metadata MUST 能记录天线数 M、子载波数 K、是否窄带选择和复数实虚部 layout

#### Scenario: 查询 LiDAR 点云 profile
- **WHEN** 开发者查询 `lidar` 模态的 `point_cloud_xyz_10000` profile
- **THEN** 系统 MUST 返回 sample key `lidar`
- **AND** 系统 MUST 返回输入语义为点云 `[T, P, 3]`
- **AND** 默认点数 P MUST 记录为 10000
- **AND** DeepSense6G BEV 和 Raymobtime occupancy grid 行为 MUST 继续通过各自 profile 或既有配置保留

### Requirement: profile 列表标准化
系统 MUST 能基于 dataset descriptor 和用户配置标准化启用模态对应的 input profiles。标准化 MUST 在 metadata 中记录每个模态的 resolved profile。

#### Scenario: Multimodal-NF 默认 profile
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf` 并启用 `image`、`lidar`、`gps` 和 `csi`
- **THEN** 系统 MUST 将 image profile 解析为 `rgb_imagenet`
- **AND** 系统 MUST 将 lidar profile 解析为 `point_cloud_xyz_10000`
- **AND** 系统 MUST 将 gps profile 解析为 `uav_xyz_snapshot`
- **AND** 系统 MUST 将 csi profile 解析为 `xl_mimo_nf`

#### Scenario: 拒绝未知 profile
- **WHEN** 用户为 Multimodal-NF 配置 `csi_profile: unknown_channel`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知 profile、模态名和可用 profile 列表

### Requirement: profile 驱动 batch 输入准备
训练、验证、评估和诊断路径 MUST 使用标准化后的 input profile 决定 batch shape 校验和必要转换。新增 profile 时，系统 MUST 不要求在每个训练循环复制 dataset-specific 分支。

#### Scenario: 准备 Multimodal-NF CSI batch
- **WHEN** batch 包含 `csi` profile `xl_mimo_nf`
- **THEN** runtime MUST 构造 `csi_batch`
- **AND** `csi_batch` MUST 保留 `[B, T, M, K, 2]` 或模型配置声明的等价 channel tensor 语义
- **AND** 缺失 `csi` 字段时 MUST 报出包含 profile 和模态名的清晰错误

#### Scenario: 准备 Multimodal-NF LiDAR 点云 batch
- **WHEN** batch 包含 `lidar` profile `point_cloud_xyz_10000`
- **THEN** runtime MUST 构造 `lidar_batch`
- **AND** `lidar_batch` MUST 保留 `[B, T, P, 3]` 点云语义，或通过配置声明的 adapter 转换为 BEV/voxel
- **AND** 未声明转换 adapter 时，模型 MUST 只接受支持点云 profile 的 encoder

#### Scenario: 旧配置不启用新 profile
- **WHEN** 用户加载现有 DeepSense6G、MMW、Raymobtime 或 CSI 配置
- **THEN** 系统 MUST 不自动设置 `uav_xyz_snapshot`、`xl_mimo_nf` 或 `point_cloud_xyz_10000`
- **AND** 旧配置的样本字段和 batch 输入准备 MUST 保持兼容
