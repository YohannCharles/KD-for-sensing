## MODIFIED Requirements

### Requirement: 模态 profile 契约
中心化模态契约 MUST 支持当前保留 dataset-specific input profile，用于在不新增模态名称的情况下表达同一模态在不同数据集中的输入语义、shape、默认字段和 batch 准备规则。profile 标准化 MUST 拒绝未知 profile，并 MUST 保持未配置旧 profile 时的既有行为。Multimodal-NF 专属 `uav_xyz_snapshot`、`xl_mimo_nf` 和 `point_cloud_xyz_10000` profiles MUST 不再作为支持 profile 保留。

#### Scenario: DeepSense/MMW/Raymobtime profile 保留
- **WHEN** 开发者查询当前保留数据集的 image、GPS、LiDAR、mmWave、CSI、coord 或 ray profile
- **THEN** 系统 MUST 返回对应 sample key、fusion input key 和输入语义
- **AND** 查询 MUST 不要求 Multimodal-NF profile 存在

#### Scenario: Multimodal-NF profile 被拒绝
- **WHEN** 用户配置 `uav_xyz_snapshot`、`xl_mimo_nf` 或 `point_cloud_xyz_10000`
- **THEN** 系统 MUST 拒绝该 profile 或因 dataset type 已退役而失败
- **AND** 错误信息 MUST 包含 profile 名称和当前可用 profile 列表

### Requirement: profile 列表标准化
系统 MUST 能基于当前保留 dataset descriptor 和用户配置标准化启用模态对应的 input profiles。标准化 MUST 在 metadata 中记录每个模态的 resolved profile。系统 MUST 不再为 `data.dataset.type: multimodal_nf` 解析默认 profile。

#### Scenario: 保留 dataset 默认 profile
- **WHEN** 用户配置当前保留 dataset 并启用多个模态
- **THEN** 系统 MUST 解析这些模态在该 dataset 下的默认或显式 profile
- **AND** metadata MUST 记录 resolved profile

#### Scenario: Multimodal-NF 默认 profile 删除
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** 系统 MUST 不解析 image/lidar/gps/csi 的 Multimodal-NF 默认 profile
- **AND** 系统 MUST 报告该 dataset type 已退役

### Requirement: profile 驱动 batch 输入准备
训练、验证、评估和诊断路径 MUST 使用标准化后的当前保留 input profile 决定 batch shape 校验和必要转换。新增 profile 时，系统 MUST 不要求在每个训练循环复制 dataset-specific 分支。Multimodal-NF CSI 和 LiDAR 点云 batch 准备不再作为支持路径。

#### Scenario: 保留 profile batch 输入
- **WHEN** batch 包含当前保留 profile 的模态字段
- **THEN** runtime MUST 构造对应 input batch
- **AND** shape 校验和缺失字段错误 MUST 使用该 profile 的语义

#### Scenario: Multimodal-NF batch 输入删除
- **WHEN** batch 或配置请求 Multimodal-NF `xl_mimo_nf` CSI batch 或 `point_cloud_xyz_10000` LiDAR batch
- **THEN** runtime MUST 不构造这些 batch 输入
- **AND** 系统 MUST 报告 profile 或 dataset type 不受支持
