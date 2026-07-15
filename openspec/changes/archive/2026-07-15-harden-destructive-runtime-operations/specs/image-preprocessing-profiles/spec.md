## MODIFIED Requirements

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
