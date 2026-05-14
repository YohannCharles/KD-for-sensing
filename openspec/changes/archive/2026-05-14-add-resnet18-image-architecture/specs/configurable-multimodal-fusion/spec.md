## ADDED Requirements

### Requirement: Fusion 支持 RGB image profile
包含 image modality 的 fusion 配置 MUST 显式或隐式携带 image profile。默认 image profile MUST 为 `rgb_imagenet`；模块化 fusion 或 ResNet-18 fusion 配置 MUST 默认使用 `rgb_imagenet`，并让 dataset、batch 准备和 image encoder 使用同一个 profile。

#### Scenario: 模块化 fusion 使用 RGB image
- **WHEN** 用户运行模块化 fusion 配置且默认或设置 `image_profile: rgb_imagenet`
- **THEN** dataset MUST 返回 RGB/ImageNet 标准化 image tensor
- **AND** image encoder MUST 接收 3 通道 RGB 输入
- **AND** 其它启用模态的输入准备语义 MUST 保持不变

### Requirement: Fusion image encoder 与 profile 校验
Fusion 模型构建 MUST 校验启用 image modality 时的 image encoder 和 image profile 是否匹配。该校验 MUST 覆盖 fusion、CRAF/MARF 风格 fusion 和新的模块化 fusion 入口，或在不支持某配置的入口处给出明确错误。

#### Scenario: fusion 使用 RGB profile
- **WHEN** 用户为 `fusion_teacher`、`fusion_student`、CRAF、MARF 或 token transformer fusion 配置 `image_profile: rgb_imagenet`
- **THEN** 系统 MUST 构建或要求 3 通道 image branch
- **AND** 错误信息 MUST 在通道数不匹配时说明期望和实际通道数

#### Scenario: ResNet-18 fusion 使用 RGB profile
- **WHEN** 用户在 fusion 中选择 ResNet-18 image encoder 且 image profile 为 `rgb_imagenet`
- **THEN** 系统 MUST 构建或运行该配置
- **AND** image batch MUST 具有 3 通道 RGB/ImageNet 输入

### Requirement: Modular fusion 复用现有模态选择语义
新的模块化 fusion 入口 MUST 复用现有 `modalities` 校验、固定模态顺序和 batch 输入字段语义。未启用的模态 MUST 不被 dataset、batch 准备、encoder 或 core 要求存在。

#### Scenario: 模块化 fusion 只启用 image 和 gps
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "gps"]`
- **THEN** batch 准备 MUST 只构造 `image_batch` 和 `gps_batch`
- **AND** 模型 forward MUST 不要求 radar、LiDAR 或 mmWave 输入

#### Scenario: 模块化 fusion 启用全部模态
- **WHEN** 模块化 fusion 配置的 `modalities` 为 `["image", "radar", "gps", "lidar", "mmwave"]`
- **THEN** 系统 MUST 为五个模态构建 encoder 和 projector
- **AND** representation core 接收的模态顺序 MUST 遵循模态契约固定顺序
