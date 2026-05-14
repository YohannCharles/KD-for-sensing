## ADDED Requirements

### Requirement: ResNet-18 ImageNet image encoder
系统 MUST 提供可注册、可配置的 ResNet-18 image encoder，用于处理 `rgb_imagenet` profile 的 RGB 帧。该 encoder MUST 基于 ResNet-18 backbone，支持 ImageNet 预训练权重，移除分类层，并将每帧编码为固定维度 embedding。

#### Scenario: 构建预训练 ResNet-18 encoder
- **WHEN** 用户在模型配置中选择 ResNet-18 ImageNet image encoder
- **THEN** 系统 MUST 构建可处理 `[B, T, 3, 224, 224]` 输入的 encoder
- **AND** encoder MUST 输出 `[B, T, D]` 帧级 embedding
- **AND** `D` MUST 由配置中的输出维度或投影层目标维度决定

#### Scenario: 缺少 torchvision 依赖
- **WHEN** 用户选择 ResNet-18 encoder 但运行环境缺少 torchvision 或对应预训练权重接口不可用
- **THEN** 系统 MUST 在构建阶段抛出清晰错误
- **AND** 错误信息 MUST 指出需要在 `kd_mm_beam` 环境中安装或修复 torchvision

### Requirement: ResNet-18 预训练与微调策略
ResNet-18 encoder MUST 支持配置预训练权重来源和训练策略。训练策略 MUST 至少支持冻结 backbone、只训练投影层、解冻后若干 stage，以及全量微调；默认策略 MUST 保守，避免小样本场景下立即全量微调导致过拟合。

#### Scenario: 冻结 backbone
- **WHEN** 用户配置 `freeze_backbone: true`
- **THEN** ResNet-18 backbone 参数 MUST 设置为不参与梯度更新
- **AND** 投影层和后续 sequence model/head 参数 MUST 仍可训练

#### Scenario: 选择性解冻 stage
- **WHEN** 用户配置解冻 ResNet-18 的后若干 stage
- **THEN** 系统 MUST 只让指定 stage、投影层和后续模块参与训练
- **AND** 系统 MUST 在模型或训练 metadata 中记录实际可训练的 ResNet-18 stage

### Requirement: ResNet-18 encoder 输出契约
ResNet-18 encoder MUST 遵守项目统一 encoder 输出契约。输入 batch 的 batch/time 维度 MUST 原样保留，输出 MUST 是 `[B, T, D]`，不得在 encoder 内执行 beam 分类、时间建模或多任务 head 逻辑。

#### Scenario: encoder 不改变时间长度
- **WHEN** 输入 image tensor 的时间维长度为 `T`
- **THEN** ResNet-18 encoder 输出的时间维长度 MUST 仍为 `T`
- **AND** 后续 GRU、Transformer、CRAF 或 MARF core MUST 接收相同时间维的 embedding

#### Scenario: encoder 不返回 logits
- **WHEN** 直接调用 ResNet-18 image encoder
- **THEN** 返回值 MUST 是帧级 feature tensor 或包含帧级 feature 的结构
- **AND** encoder MUST 不直接返回 beam logits

### Requirement: ResNet-18 是 RGB/ImageNet 默认 encoder
系统 MUST 将 ResNet-18 ImageNet encoder 作为 `rgb_imagenet` 默认/推荐 image encoder。包含 image 的默认配置 MUST 使用 RGB/ImageNet 输入契约；旧单通道 image encoder 名称 MUST 被拒绝。

#### Scenario: 默认 RGB 路径选择 ResNet-18
- **WHEN** 用户运行未绑定 legacy motion branch 的默认 image 配置
- **THEN** 系统 MUST 使用 `rgb_imagenet` profile 和 ResNet-18 image encoder
- **AND** 输出 logits MUST 继续兼容现有 beam prediction 训练、评估和 distillation 流程

#### Scenario: 显式选择 ResNet-18
- **WHEN** 用户运行新的 ResNet-18 image-only 或 modular fusion 配置
- **THEN** 系统 MUST 使用 `rgb_imagenet` profile 和 ResNet-18 image encoder
- **AND** 输出 logits MUST 继续兼容现有 beam prediction 训练、评估和 distillation 流程
