## ADDED Requirements

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
