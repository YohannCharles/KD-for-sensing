# modality-contracts Specification

## Purpose
定义中心化模态顺序、dataset flag、batch key 和默认字段推导契约，确保配置、数据集、模型和诊断共享同一模态语义。
## Requirements
### Requirement: 中心化模态契约
项目 MUST 提供单一来源的模态契约，用于描述所有受支持模态的规范名称、固定顺序、dataset flag、样本字段、fusion 输入字段、默认 dataset/model 字段，以及是否支持 cache 或归一化 artifact。image modality MUST 不暴露 image motion cache、motion profile 或 motion encoder 推荐。

#### Scenario: 枚举受支持模态
- **WHEN** 开发者查询模态契约
- **THEN** 系统 MUST 返回固定顺序的 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi`
- **AND** 该顺序 MUST 被 canonical config、fusion 模态解析、dataset 构建和诊断配置复用

#### Scenario: 查询 image 模态元数据
- **WHEN** 开发者查询 `image` 模态契约
- **THEN** 系统 MUST 返回 image 对应的样本字段 `image`
- **AND** 系统 MUST 返回 fusion 输入字段 `image_batch`
- **AND** 系统 MUST 返回 RGB/ImageNet 输入契约
- **AND** 系统 MUST 不返回 image motion cache 能力

#### Scenario: 查询 radar 模态元数据
- **WHEN** 开发者查询 `radar` 模态契约
- **THEN** 系统 MUST 返回 radar 对应的样本字段 `radar_ra` 和 `radar_da`
- **AND** 系统 MUST 返回 fusion 输入字段 `radar_batch`

#### Scenario: 查询 CSI 模态元数据
- **WHEN** 开发者查询 `csi` 模态契约
- **THEN** 系统 MUST 返回 CSI 对应的样本字段 `csi`
- **AND** 系统 MUST 返回 fusion 输入字段 `csi_batch`
- **AND** 系统 MUST 返回 dataset flag `use_csi`
- **AND** 系统 MUST 返回 CSI RMS normalizer artifact key

### Requirement: 模态列表标准化
系统 MUST 通过模态契约标准化用户配置中的模态列表。标准化 MUST 拒绝未知模态、空列表和重复模态，并 MUST 按固定模态顺序返回结果。

#### Scenario: 标准化乱序 fusion 模态
- **WHEN** 用户配置 fusion `modalities: ["csi", "lidar", "image", "gps"]`
- **THEN** 系统 MUST 将有效模态标准化为 `["image", "gps", "lidar", "csi"]`
- **AND** dataset、batch 准备和模型构建 MUST 使用同一个标准化结果

#### Scenario: 拒绝未知模态
- **WHEN** 用户配置 `modalities: ["image", "thermal"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知模态和可用模态列表

#### Scenario: 拒绝重复模态
- **WHEN** 用户配置 `modalities: ["csi", "csi"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出模态列表包含重复项

### Requirement: 模态契约驱动 dataset flag
数据构建流程 MUST 通过模态契约生成 dataset flag 和相关默认字段，避免在多个模块中手写 `use_gps`、`use_lidar`、`use_mmwave`、`use_csi` 等分支。

#### Scenario: GPS、mmWave 与 CSI fusion 设置 dataset flag
- **WHEN** fusion 启用模态为 `["gps", "mmwave", "csi"]`
- **THEN** 数据构建流程 MUST 设置 `use_gps: true`、`use_mmwave: true` 和 `use_csi: true`
- **AND** 数据构建流程 MUST 不设置与未启用 LiDAR 相关的启用 flag

#### Scenario: 单模态 image 不启用可选模态 flag
- **WHEN** `experiment.task: image`
- **THEN** 数据构建流程 MUST 只启用 image 所需字段
- **AND** GPS、LiDAR、mmWave 和 CSI 的 dataset flag MUST 保持关闭或缺省关闭

### Requirement: 模态契约驱动 batch 输入
训练、验证、评估和诊断路径 MUST 使用模态契约确定 batch 字段到模型输入参数的映射。新增或调整模态输入键时，系统 MUST 不要求在训练、验证和评估循环中复制分支逻辑。

#### Scenario: fusion batch 输入映射
- **WHEN** fusion 启用模态为 `["radar", "mmwave", "csi"]`
- **THEN** batch 准备流程 MUST 从样本字段构建 `radar_batch`、`mmwave_batch` 和 `csi_batch`
- **AND** 模型调用 MUST 不传入未启用 image、GPS 或 LiDAR 的输入张量

#### Scenario: 单模态 batch 输入映射
- **WHEN** `experiment.task: csi`
- **THEN** batch 准备流程 MUST 构建 CSI 模型所需输入
- **AND** 训练和评估循环 MUST 通过统一映射调用模型

### Requirement: 模态契约文档化
项目文档 MUST 说明模态契约的职责，以及新增模态时必须补充的 dataset 字段、batch 字段、model 输入、cache/normalization 能力和测试。

#### Scenario: 按文档新增模态
- **WHEN** 开发者按照扩展文档新增一个实验性模态
- **THEN** 文档 MUST 指引开发者先新增模态契约
- **AND** 文档 MUST 指出后续需要补充 dataset 读取、模型注册、batch 准备和诊断显示逻辑

### Requirement: RGB image profile 元数据
模态契约 MUST 为 image modality 暴露 RGB/ImageNet 输入 profile 元数据。元数据 MUST 至少包含 profile 名称、期望通道数、默认空间尺寸、dataset 样本字段、fusion 输入字段、是否支持 cache、以及推荐 encoder 类型。

#### Scenario: 查询 RGB ImageNet profile 元数据
- **WHEN** 开发者查询 image modality 的 `rgb_imagenet` profile
- **THEN** 系统 MUST 返回通道数 3、默认空间尺寸 224x224、样本字段 `image`、fusion 输入字段 `image_batch`
- **AND** 系统 MUST 标记该 profile 不支持 image cache
- **AND** 系统 MUST 推荐 `resnet18_imagenet_rgb` encoder

### Requirement: Image profile 标准化
系统 MUST 通过模态契约或等价中心化函数标准化 `image_profile` 配置。标准化 MUST 拒绝未知或已删除 profile，并 MUST 为未配置 profile 的默认路径返回 `rgb_imagenet`。

#### Scenario: 默认配置标准化
- **WHEN** 用户配置启用 image modality 且未设置 `image_profile`
- **THEN** 标准化结果 MUST 为 `rgb_imagenet`
- **AND** dataset、batch 准备和模型构建 MUST 使用同一个标准化结果

#### Scenario: RGB 配置标准化
- **WHEN** 用户配置 `image_profile: rgb_imagenet`
- **THEN** 标准化结果 MUST 保留为 `rgb_imagenet`
- **AND** 后续配置校验 MUST 能据此要求 3 通道 RGB encoder

### Requirement: Batch 输入准备使用 RGB image profile
训练、验证、评估和诊断路径 MUST 使用标准化后的 image profile 决定 image batch 准备逻辑。batch 准备 MUST 在进入模型前形成明确的 `[B, T, 3, H, W]` tensor，并 MUST 使用统一的历史长度和 future padding 策略。

#### Scenario: RGB batch 准备
- **WHEN** image profile 为 `rgb_imagenet`
- **THEN** batch 准备 MUST 接受 dataset 返回的 RGB 帧序列
- **AND** 传给模型的通道数 MUST 为 3
- **AND** future padding MUST 不改变历史 RGB 帧的标准化值

### Requirement: Image 模态仅支持 RGB/ImageNet 输入
系统 MUST 将 image modality 的输入契约固定为 RGB/ImageNet 路径。配置解析、模态契约和模型构建 MUST 拒绝 `motion_mask` profile、motion cache 能力和 motion image encoder。

#### Scenario: 默认 image profile 为 RGB/ImageNet
- **WHEN** 开发者查询 image modality 的输入契约
- **THEN** 系统 MUST 返回 RGB/ImageNet 输入语义
- **AND** 系统 MUST 返回 3 通道、224x224 的默认空间尺寸
- **AND** 系统 MUST 标记该 image 输入不使用 image motion cache

#### Scenario: motion profile 不可解析
- **WHEN** 用户配置 `image_profile: motion_mask`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含 `motion_mask` 已删除和可用 image 输入契约

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

### Requirement: Difficulty profile 复用 canonical modality keys
Difficulty profile MUST 使用中心化模态契约中的 canonical modality name、sample key 和 fusion input key 来声明 affected modality。难度 profile MUST 不新增 `gps_noisy`、`delayed_gps`、`image_hard` 等伪模态名称，也 MUST 不要求训练、评估或模型 forward 为每种难度新增专用输入分支。

#### Scenario: GPS difficulty 使用 gps canonical key
- **WHEN** profile 声明 GPS jitter、delay 或 dropout
- **THEN** affected modality MUST 标准化为 `gps`
- **AND** transform MUST 作用于当前 batch 的 GPS sample key，并由现有 `gps_batch` 准备路径消费

#### Scenario: 拒绝伪模态名称
- **WHEN** 用户在 modalities 或 difficulty affected modality 中配置 `delayed_gps`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指向 canonical modality `gps` 和 difficulty profile 配置

### Requirement: Difficulty mask 与 metadata 字段语义
模态契约或等价中心化 helper MUST 定义 difficulty 产生的输入相关 mask/metadata 字段语义。GPS async/missing 字段至少 MUST 覆盖 valid、stale、delay steps、source index 和 dropout mask；image degradation 字段至少 MUST 覆盖 degradation type、severity、seed、frame range 和 optional mask。字段命名 MUST 避免与 target schema、auxiliary target 或 sensitive supervision 字段混淆。

#### Scenario: GPS async metadata 可查询
- **WHEN** 开发者查询 GPS modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 `gps_valid_mask`、`gps_stale_mask`、`gps_delay_steps`、`gps_source_index` 和 `gps_dropout_mask` 或等价字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision

#### Scenario: image degradation metadata 不改变 profile
- **WHEN** image difficulty operator 输出 degradation metadata
- **THEN** metadata MUST 记录为 difficulty metadata
- **AND** image input profile MUST 仍保持 `rgb_imagenet` 或配置解析后的当前 profile

### Requirement: Image observability metadata 字段
模态契约或等价中心化 helper MUST 定义 image observability 相关 difficulty metadata 字段语义。字段至少 MUST 覆盖 `image_valid_mask`、`image_observability_score`、`image_dropout_mask`、`image_burst_dropout_mask`、`image_degradation_metadata`、corruption type、severity、seed 和 frame range。

#### Scenario: 查询 image observability metadata
- **WHEN** 开发者查询 image modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 image valid mask、observability score、dropout/burst masks 和 degradation metadata 的字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision 或辅助标签

#### Scenario: metadata 字段不创建伪模态
- **WHEN** 配置启用 Scenario D image observability difficulty
- **THEN** affected modality MUST 仍标准化为 canonical `image`
- **AND** 系统 MUST 拒绝 `image_hard`、`missing_image_modality` 或其它伪模态名称

### Requirement: Reliability metadata 进入 batch 输入映射
训练、评估和 benchmark batch 输入映射 MUST 能将 image/GPS reliability metadata 传递给显式支持的模型，同时保持不支持该 metadata 的模型兼容。metadata 传递 MUST 不要求每个 difficulty condition 新增专用模型输入分支。

#### Scenario: observability-aware 模型接收 metadata
- **WHEN** 模型配置声明需要 observability-aware fusion
- **THEN** batch 准备 MUST 向模型 forward 提供 image observability 和 GPS reliability metadata
- **AND** 缺少字段时 MUST 抛出清晰错误或记录配置声明的 fallback warning

#### Scenario: legacy-compatible baseline 忽略 metadata
- **WHEN** standard Image ResNet+GPS 或 Image-AE+GPS baseline 不声明 reliability metadata 输入
- **THEN** batch 准备 MUST 允许其忽略 Scenario D metadata
- **AND** benchmark comparability metadata MUST 记录该模型未消费 reliability metadata

