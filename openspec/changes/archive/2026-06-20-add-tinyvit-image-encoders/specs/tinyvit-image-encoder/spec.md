## ADDED Requirements

### Requirement: TinyViT RGB image encoder versions
系统 MUST 提供 TinyViT-5M 和 TinyViT-11M 作为可注册、可配置的 RGB/ImageNet image encoder。系统 MUST 暴露四个 opt-in encoder 注册名：`tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb` 和 `tinyvit_11m_22k_rgb`。这些 encoder MUST 作为 `ENCODERS` 组件使用，并 MUST 不替换当前默认 `resnet18_imagenet_rgb`。

#### Scenario: 构建 TinyViT-5M scratch encoder
- **WHEN** 用户在 `model.primary.encoders.image.type` 中选择 `tinyvit_5m_scratch_rgb`
- **THEN** 系统 MUST 构建 TinyViT-5M image encoder
- **AND** 系统 MUST 不下载或加载 22k 预训练权重
- **AND** encoder MUST 可被 `modular_sequence` 作为 image encoder 使用

#### Scenario: 构建 TinyViT-5M 22k encoder
- **WHEN** 用户在 `model.primary.encoders.image.type` 中选择 `tinyvit_5m_22k_rgb`
- **THEN** 系统 MUST 构建 TinyViT-5M image encoder
- **AND** 系统 MUST 使用 ImageNet-22k distill 预训练权重加载策略
- **AND** encoder MUST 可被 `modular_sequence` 作为 image encoder 使用

#### Scenario: 构建 TinyViT-11M scratch encoder
- **WHEN** 用户在 `model.primary.encoders.image.type` 中选择 `tinyvit_11m_scratch_rgb`
- **THEN** 系统 MUST 构建 TinyViT-11M image encoder
- **AND** 系统 MUST 不下载或加载 22k 预训练权重
- **AND** encoder MUST 可被 `modular_sequence` 作为 image encoder 使用

#### Scenario: 构建 TinyViT-11M 22k encoder
- **WHEN** 用户在 `model.primary.encoders.image.type` 中选择 `tinyvit_11m_22k_rgb`
- **THEN** 系统 MUST 构建 TinyViT-11M image encoder
- **AND** 系统 MUST 使用 ImageNet-22k distill 预训练权重加载策略
- **AND** encoder MUST 可被 `modular_sequence` 作为 image encoder 使用

#### Scenario: 默认 image 配置不切换 TinyViT
- **WHEN** 用户加载当前 canonical image 或包含 image 的 fusion 配置且未显式选择 TinyViT
- **THEN** 系统 MUST 继续使用当前默认 image encoder
- **AND** 系统 MUST 不自动把 `resnet18_imagenet_rgb` 替换为 TinyViT

### Requirement: TinyViT encoder 输入输出契约
TinyViT image encoder MUST 遵守项目统一 encoder 契约。encoder MUST 接收 `rgb_imagenet` profile 的 `[B, T, 3, 224, 224]` image batch，并 MUST 输出 `[B, T, D]` 帧级 embedding。TinyViT encoder MUST 不在 encoder 内执行 beam 分类、时间建模、fusion 或 head 逻辑。

#### Scenario: TinyViT forward 输出帧级 embedding
- **WHEN** TinyViT image encoder 接收形状为 `[B, T, 3, 224, 224]` 的 image batch
- **THEN** encoder MUST 返回形状为 `[B, T, D]` 的 tensor
- **AND** `D` MUST 由 `output_dim`、`feature_size`、`d_model` 或默认投影维度决定
- **AND** 输出时间维 MUST 与输入 `T` 一致

#### Scenario: TinyViT 拒绝错误 image profile
- **WHEN** 用户将 TinyViT image encoder 与非 `rgb_imagenet` image profile 或非 3 通道输入组合
- **THEN** 系统 MUST 在构建或首次 forward 时拒绝该配置
- **AND** 错误信息 MUST 包含 encoder 名称、image profile、期望通道数和实际通道数

#### Scenario: TinyViT 拒绝错误 image shape
- **WHEN** TinyViT image encoder 收到非 `[B, T, 3, 224, 224]` 的 image tensor
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 包含实际输入 shape 和 TinyViT 需要的 shape

#### Scenario: TinyViT encoder 不返回 ImageNet logits
- **WHEN** 直接调用 TinyViT image encoder
- **THEN** 返回值 MUST 是帧级 feature tensor
- **AND** encoder MUST 不直接返回 ImageNet 分类 logits 或 beam logits

### Requirement: TinyViT 22k 预训练权重加载
TinyViT 22k 预训练版本 MUST 支持可审计的权重加载策略。系统 MUST 支持本地 checkpoint 路径优先，并 MAY 在用户显式选择 22k 版本且允许下载时从上游 URL 获取权重。系统 MUST 不提交、复制或生成预训练权重到源码变更中。

#### Scenario: 使用本地 TinyViT checkpoint
- **WHEN** 用户为 TinyViT 22k encoder 提供 `checkpoint_path`
- **THEN** 系统 MUST 从该路径加载权重
- **AND** 系统 MUST 不访问网络下载权重
- **AND** metadata MUST 记录本地 checkpoint 路径和 checkpoint schema

#### Scenario: 使用上游 TinyViT 22k 权重
- **WHEN** 用户选择 TinyViT 22k encoder 且未提供 `checkpoint_path`
- **THEN** 系统 MUST 使用该 variant 对应的 ImageNet-22k distill checkpoint URL 或 torch hub cache 策略
- **AND** 下载或缓存失败时 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出可提供本地 `checkpoint_path`

#### Scenario: scratch TinyViT 不加载预训练权重
- **WHEN** 用户选择 `tinyvit_5m_scratch_rgb` 或 `tinyvit_11m_scratch_rgb`
- **THEN** 系统 MUST 使用随机初始化权重
- **AND** 系统 MUST 不下载 TinyViT checkpoint
- **AND** metadata MUST 记录 `pretrained: false`

#### Scenario: 过滤 downstream 不需要的分类权重
- **WHEN** 系统加载 TinyViT ImageNet-22k distill checkpoint
- **THEN** 系统 MUST 过滤或忽略 downstream encoder 不需要的 ImageNet 分类 head 权重
- **AND** 系统 MUST 过滤上游非持久 attention index buffer
- **AND** 其它 unexpected 或 shape mismatch 权重 MUST 触发清晰错误

### Requirement: TinyViT 微调策略 metadata
TinyViT image encoder MUST 支持可配置训练策略，并 MUST 通过 `training_strategy_metadata()` 或等价 metadata 暴露实际策略。默认策略 MUST 保守冻结 TinyViT backbone，只训练投影层和后续模块；显式配置 MAY 解冻全部 backbone 或解冻指定 stage。

#### Scenario: 默认冻结 TinyViT backbone
- **WHEN** 用户未显式配置 TinyViT 微调策略
- **THEN** TinyViT backbone 参数 MUST 不参与梯度更新
- **AND** projection、projector、representation core 和 head 参数 MUST 仍可训练
- **AND** metadata MUST 记录 `freeze_backbone: true`

#### Scenario: 全量微调 TinyViT backbone
- **WHEN** 用户配置 `freeze_backbone: false`
- **THEN** TinyViT backbone 参数 MUST 参与梯度更新
- **AND** metadata MUST 记录全量微调策略

#### Scenario: 选择性解冻 TinyViT stage
- **WHEN** 用户配置 `unfreeze_stages` 或 `unfreeze_last_n_stages`
- **THEN** 系统 MUST 只让声明的 TinyViT stage 参与梯度更新
- **AND** 系统 MUST 拒绝未知 stage 名称
- **AND** metadata MUST 记录实际 `trainable_stages`

#### Scenario: TinyViT metadata 被 modular_sequence 聚合
- **WHEN** `modular_sequence` 使用 TinyViT image encoder 完成构建或训练
- **THEN** run metadata MUST 记录 image encoder 类型、TinyViT variant、预训练来源、权重路径或 URL、freeze policy、trainable stages、backbone_dim 和 output_dim
- **AND** 普通 TinyViT baseline MUST 标记为不消费 reliability metadata
