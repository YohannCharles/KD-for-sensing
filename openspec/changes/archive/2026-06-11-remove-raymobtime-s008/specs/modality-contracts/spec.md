## REMOVED Requirements

### Requirement: Raymobtime snapshot 模态契约
**Reason**: `coord` 和 `ray` 是 Raymobtime s008 工作流引入的专属模态扩展；该工作流退役后不再作为中心化模态契约的一部分维护。
**Migration**: 当前保留 workflow 继续使用 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi`；如需新的坐标或 ray-tracing 模态，必须提出新的 OpenSpec change。

#### Scenario: coord/ray 不再枚举为当前支持模态
- **WHEN** 开发者查询受支持模态
- **THEN** 系统 MUST 不返回 `coord` 或 `ray` 作为当前支持模态
- **AND** 既有保留模态顺序 MUST 保持稳定

### Requirement: Raymobtime 模态列表标准化
**Reason**: Raymobtime s008 已退役，包含 `coord` 或 `ray` 的 Raymobtime 模态列表不再需要标准化。
**Migration**: 使用当前保留模态；包含 `coord` 或 `ray` 的旧 Raymobtime 配置必须失败。

#### Scenario: coord/ray 模态配置被拒绝
- **WHEN** 用户配置 `modalities: ["coord"]`、`modalities: ["ray"]` 或包含二者的 Raymobtime s008 模态列表
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出未知模态或 Raymobtime s008 已退役

### Requirement: coord/ray batch 输入准备
**Reason**: `coord_batch` 和 `ray_batch` 准备逻辑只服务 Raymobtime s008 snapshot 模型；该模型退役后不再需要 runtime 支持。
**Migration**: 当前保留 workflow 使用各自模态 profile 的 batch 准备；不迁移 `coord/ray` batch 字段。

#### Scenario: runtime 不构造 coord/ray batch
- **WHEN** batch 或配置请求 `coord_batch` 或 `ray_batch`
- **THEN** runtime MUST 不构造这些输入
- **AND** 系统 MUST 报告对应模态不受支持或 Raymobtime s008 已退役

## MODIFIED Requirements

### Requirement: 模态 profile 契约
中心化模态契约 MUST 支持当前保留 dataset-specific input profile，用于在不新增模态名称的情况下表达同一模态在不同数据集中的输入语义、shape、默认字段和 batch 准备规则。profile 标准化 MUST 拒绝未知 profile，并 MUST 保持未配置旧 profile 时的既有行为。Multimodal-NF 专属 `uav_xyz_snapshot`、`xl_mimo_nf` 和 `point_cloud_xyz_10000` profiles MUST 不再作为支持 profile 保留，Raymobtime s008 专属 profile MUST 不再作为当前保留 profile。

#### Scenario: 保留数据集 profile 可查询
- **WHEN** 开发者查询当前保留数据集的 image、GPS、LiDAR、mmWave 或 CSI profile
- **THEN** 系统 MUST 返回对应 sample key、fusion input key 和输入语义
- **AND** 查询 MUST 不要求 Multimodal-NF 或 Raymobtime s008 profile 存在

#### Scenario: Multimodal-NF profile 被拒绝
- **WHEN** 用户配置 `uav_xyz_snapshot`、`xl_mimo_nf` 或 `point_cloud_xyz_10000`
- **THEN** 系统 MUST 拒绝该 profile 或因 dataset type 已退役而失败
- **AND** 错误信息 MUST 包含 profile 名称和当前可用 profile 列表

#### Scenario: Raymobtime profile 被拒绝
- **WHEN** 用户配置 Raymobtime s008 专属 coord、ray 或 LiDAR occupancy profile
- **THEN** 系统 MUST 拒绝该 profile 或因 Raymobtime s008 已退役而失败
- **AND** 错误信息 MUST 包含 Raymobtime s008 已退役或当前可用 profile 列表
