# image-preprocessing-profiles Specification

## Purpose
定义 image 输入 profile、RGB/ImageNet 预处理、可选 image-derived cache，以及已退役 image motion cache 的边界。
## Requirements
### Requirement: Image 预处理 profile 选择
系统 MUST 为 image modality 提供显式 `image_profile` 配置。`image_profile` MUST 支持 `rgb_imagenet`，默认值 MUST 为 `rgb_imagenet`；已删除的旧 image profile MUST 被拒绝。RGB/ImageNet image-derived cache MAY 作为显式 cache policy 控制的等价加速路径，但 MUST 不改变 profile 输出契约，也 MUST 不恢复 image motion cache。

#### Scenario: 默认 image profile 为 RGB/ImageNet
- **WHEN** 用户加载未设置 `image_profile` 的 image 配置
- **THEN** 系统 MUST 使用 `rgb_imagenet`
- **AND** dataset 返回的 image 字段 MUST 是 RGB/ImageNet 标准化张量
- **AND** 未启用 image-derived cache 时 MUST 直接读取原始帧，启用时命中结果 MUST 与直接变换等价

#### Scenario: 拒绝未知或已删除 image profile
- **WHEN** 用户配置未知或已删除 image profile
- **THEN** 系统 MUST 拒绝构建 dataset 或配置
- **AND** 错误信息 MUST 包含请求的 profile 名称和可用 profile 列表或迁移提示

### Requirement: RGB ImageNet profile 契约
`rgb_imagenet` profile MUST 将每帧转换为 RGB 三通道，resize 或 resize+center-crop 到 224x224，转换为 float tensor，并按 ImageNet mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]` 标准化。可选 image-derived cache MUST 以规范化相对资源 identity、image size、profile 和 transform version 区分条目。

#### Scenario: rgb_imagenet shape 与标准化
- **WHEN** dataset 在 `rgb_imagenet` profile 下加载 image modality
- **THEN** 返回样本 MUST 表示最近 `seq_len` 个 RGB 帧
- **AND** 每帧 MUST 为 3 通道 224x224，batch MUST 为 `[B, T, 3, 224, 224]`

#### Scenario: RGB profile cache 等价
- **WHEN** 同一帧分别通过直接 transform 与 fingerprint 匹配的 image-derived cache 读取
- **THEN** 两条路径的 shape、dtype 和数值语义 MUST 一致
- **AND** run metadata MUST 记录 profile、cache policy、transform version 与 hit/miss 摘要
- **AND** 任何旧 image motion cache MUST 不被读取、创建或迁移

### Requirement: Image profile 与模型匹配校验
系统 MUST 在配置解析、dataset 构建或模型构建阶段校验 image profile 与 image encoder 的输入契约。`rgb_imagenet` profile MUST 匹配 3 通道 RGB encoder；不匹配时 MUST 抛出清晰错误，而不是让卷积层 shape mismatch 暴露到训练中途。

#### Scenario: ResNet-18 使用 RGB/ImageNet profile
- **WHEN** 用户配置 ResNet-18 ImageNet encoder 且 `image_profile: rgb_imagenet`
- **THEN** 系统 MUST 构建可接收 3 通道 RGB 输入的 encoder
- **AND** 输出 MUST 与统一 `[B, T, D]` encoder 契约一致

#### Scenario: 已删除 image encoder 被拒绝
- **WHEN** 用户配置已删除的旧 image encoder 名称
- **THEN** 系统 MUST 拒绝该配置
- **AND** 错误信息 MUST 指向 `resnet18_imagenet_rgb` 和 `rgb_imagenet` 迁移路径

### Requirement: Fusion 支持 RGB image profile
包含 image modality 的 fusion 配置 MUST 显式或隐式携带 image profile。默认 image profile MUST 为 `rgb_imagenet`；模块化 fusion 或 ResNet-18 fusion 配置 MUST 默认使用 `rgb_imagenet`，并让 dataset、batch 准备和 image encoder 使用同一个 profile。

#### Scenario: 模块化 fusion 使用 RGB image
- **WHEN** 用户运行模块化 fusion 配置且默认或设置 `image_profile: rgb_imagenet`
- **THEN** dataset MUST 返回 RGB/ImageNet 标准化 image tensor
- **AND** image encoder MUST 接收 3 通道 RGB 输入
- **AND** 其它启用模态的输入准备语义 MUST 保持不变

### Requirement: Fusion image encoder 与 profile 校验
Fusion 模型构建 MUST 校验启用 image modality 时的 image encoder 和 image profile 是否匹配。该校验 MUST 覆盖当前保留的 fusion、token transformer fusion 和模块化 fusion 入口，或在不支持某配置的入口处给出明确错误。

#### Scenario: fusion 使用 RGB profile
- **WHEN** 用户为 `fusion_teacher`、`fusion_student` 或 token transformer fusion 配置 `image_profile: rgb_imagenet`
- **THEN** 系统 MUST 构建或要求 3 通道 image branch
- **AND** 错误信息 MUST 在通道数不匹配时说明期望和实际通道数

#### Scenario: ResNet-18 fusion 使用 RGB profile
- **WHEN** 用户在 fusion 中选择 ResNet-18 image encoder 且 image profile 为 `rgb_imagenet`
- **THEN** 系统 MUST 构建或运行该配置
- **AND** image batch MUST 具有 3 通道 RGB/ImageNet 输入

#### Scenario: 已退役 fusion 方法不参与 profile 校验
- **WHEN** 配置请求 CRAF 或 MARF 风格 fusion
- **THEN** 系统 MUST 在 profile 校验前拒绝该模型类型
- **AND** 系统 MUST 不进入 CRAF/MARF 专属 image branch 构建逻辑
