# modality-contracts Specification

## Purpose
TBD - created by archiving change clarify-architecture-boundaries. Update Purpose after archive.
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

### Requirement: Raymobtime snapshot 模态契约
中心化模态契约 MUST 支持 Raymobtime s008 所需的 `coord` 和 `ray` 模态。新增模态 MUST 不改变既有 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi` 的相对顺序、默认字段和旧配置行为。

#### Scenario: 枚举 Raymobtime 新模态
- **WHEN** 开发者查询受支持模态
- **THEN** 系统 MUST 返回既有模态以及 `coord` 和 `ray`
- **AND** 既有六个模态的相对顺序 MUST 保持不变
- **AND** `coord` 和 `ray` MUST 只在用户显式配置或 Raymobtime 配置中启用

#### Scenario: 查询 coord 模态元数据
- **WHEN** 开发者查询 `coord` 模态契约
- **THEN** 系统 MUST 返回样本字段 `coord`
- **AND** 系统 MUST 返回 fusion 输入字段 `coord_batch`
- **AND** 系统 MUST 返回 dataset flag `use_coord`
- **AND** 系统 MUST 返回当前 snapshot coordinate 输入契约和默认 `coord_input_size`

#### Scenario: 查询 ray 模态元数据
- **WHEN** 开发者查询 `ray` 模态契约
- **THEN** 系统 MUST 返回样本字段 `ray`
- **AND** 系统 MUST 返回 fusion 输入字段 `ray_batch`
- **AND** 系统 MUST 返回 dataset flag `use_ray`
- **AND** 系统 MUST 返回 path-level ray-tracing feature 输入契约和默认 `ray_input_size`

### Requirement: Raymobtime 模态列表标准化
系统 MUST 通过中心化模态契约标准化包含 `coord` 和 `ray` 的模态列表。标准化 MUST 继续拒绝未知模态、空列表和重复模态，并 MUST 按固定顺序返回结果。

#### Scenario: 标准化 Raymobtime 全模态
- **WHEN** 用户配置 Raymobtime 模态列表 `["ray", "coord", "lidar", "image"]`
- **THEN** 系统 MUST 返回同一组模态的规范顺序
- **AND** dataset、batch 准备、模型构建和分析输出 MUST 使用同一个标准化结果

#### Scenario: 拒绝重复 Raymobtime 模态
- **WHEN** 用户配置 `modalities: ["coord", "coord", "image"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指出模态列表包含重复项

#### Scenario: 拒绝未知 Raymobtime 模态
- **WHEN** 用户配置 `modalities: ["coord", "ray_path"]`
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 包含未知模态和可用模态列表

### Requirement: coord/ray batch 输入准备
训练、验证、评估和诊断路径 MUST 能根据模态契约把 `coord` 和 `ray` batch 字段准备为模型输入。Raymobtime snapshot 输入 MUST 保留单步维度，并在模型进入前校验维度和 dtype。

#### Scenario: 准备 coord batch
- **WHEN** 模型启用 `coord` 模态且 batch 包含 `coord`
- **THEN** runtime MUST 构造 `coord_batch`
- **AND** `coord_batch` MUST 具有 `[B, 1, F_coord]` 语义
- **AND** 缺失 `coord` 字段时 MUST 报出包含当前模态名的清晰错误

#### Scenario: 准备 ray batch
- **WHEN** 模型启用 `ray` 模态且 batch 包含 `ray`
- **THEN** runtime MUST 构造 `ray_batch`
- **AND** `ray_batch` MUST 具有 `[B, 1, F_ray]` 语义
- **AND** 缺失 `ray` 字段时 MUST 报出包含当前模态名的清晰错误

#### Scenario: 旧配置不启用 coord/ray
- **WHEN** 用户加载现有 DeepSense6G、MMW 或 CSI 配置
- **THEN** 数据构建流程 MUST 不设置 `use_coord` 或 `use_ray`
- **AND** batch 准备 MUST 不要求 `coord` 或 `ray` 字段

