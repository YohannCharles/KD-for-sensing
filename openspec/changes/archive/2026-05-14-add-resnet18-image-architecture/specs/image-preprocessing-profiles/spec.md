## ADDED Requirements

### Requirement: Image 预处理 profile 选择
系统 MUST 为 image modality 提供显式 `image_profile` 配置。`image_profile` MUST 支持 `rgb_imagenet`，默认值 MUST 为 `rgb_imagenet`；已删除的旧 image profile MUST 被拒绝。

#### Scenario: 默认 image profile 为 RGB/ImageNet
- **WHEN** 用户加载未设置 `image_profile` 的 image 配置
- **THEN** 系统 MUST 使用 `rgb_imagenet`
- **AND** dataset 返回的 image 字段 MUST 是 RGB/ImageNet 标准化张量
- **AND** 系统 MUST 不读取或写入 image cache

#### Scenario: 拒绝未知或已删除 image profile
- **WHEN** 用户配置未知或已删除 image profile
- **THEN** 系统 MUST 拒绝构建 dataset 或配置
- **AND** 错误信息 MUST 包含请求的 profile 名称和可用 profile 列表或迁移提示

### Requirement: RGB ImageNet profile 契约
`rgb_imagenet` profile MUST 直接加载原始 RGB 帧并生成 ResNet-18 兼容输入。系统 MUST 将每帧转换为 RGB 三通道，resize 或 resize+center-crop 到 224x224，转换为 float tensor，并按 ImageNet mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]` 标准化。

#### Scenario: rgb_imagenet shape 与标准化
- **WHEN** dataset 在 `rgb_imagenet` profile 下加载 image modality
- **THEN** 返回的样本 image 字段 MUST 表示最近 `seq_len` 个 RGB 帧
- **AND** 每帧 MUST 具有 3 个通道和 224x224 空间尺寸
- **AND** batch 准备后传给模型的 image tensor MUST 具有 `[B, T, 3, 224, 224]` 结构

#### Scenario: RGB profile 不写 image cache
- **WHEN** `image_profile: rgb_imagenet`
- **THEN** dataset MUST 不读取、不创建、不写入 image cache 文件
- **AND** 系统 MUST 在 run metadata 或配置解析结果中记录实际使用的 `image_profile`

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
