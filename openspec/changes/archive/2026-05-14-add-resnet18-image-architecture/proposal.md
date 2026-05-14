## Why

当前 image 输入已经统一到 RGB/ImageNet 路径。Scenario 31-34 这类真实 RGB 街景需要一个可注册、可配置的 ResNet-18 image encoder，同时训练代码也需要一个模块化序列模型入口，避免为每个新 image encoder 重复复制时序建模和 head 逻辑。

## What Changes

- 新增并默认使用 `rgb_imagenet` image profile，将原始 RGB 帧 resize 到 224x224、转换为 3 通道 float tensor，并使用 ImageNet mean/std 标准化。
- 新增 ResNet-18 image encoder：支持 ImageNet 预训练、移除分类层、输出帧级 `[B, T, D]` embedding，并可配置冻结 backbone、选择性解冻 stage 或全量微调。
- 新增模块化序列模型注册入口，抽象为 `encoders -> projectors -> representation_core -> heads`，支持 image-only 与多模态 fusion。
- 为 image 输入建立显式配置与校验：RGB/ImageNet profile 必须匹配 3 通道 image encoder；已删除的旧单通道 image 路径和旧 encoder 名称必须被拒绝。
- 新增 canonical/example 配置，覆盖 `resnet18_imagenet_rgb` image-only baseline，以及可用于 fusion/core 对比的 modular sequence 配置。
- 补充测试：RGB/ImageNet shape/range、ResNet-18 encoder 输出 shape、错误配置校验、注册构建和最小 forward 回归。

## Capabilities

### New Capabilities

- `image-preprocessing-profiles`: 定义 RGB/ImageNet image profile、输入张量契约和配置校验。
- `resnet18-image-encoder`: 定义可注册的 ResNet-18 ImageNet image encoder、输出契约和预训练/冻结策略。
- `modular-sequence-model`: 定义 encoder/projector/representation core/head 可插拔的序列模型架构，用于 image-only 和多模态 fusion 的新增实验入口。

### Modified Capabilities

- `component-registry`: 新增 image encoder、representation core、head 或模块化模型的注册与构建要求，并保持 registry 轻量导入边界。
- `modality-contracts`: 为 image 模态补充 RGB/ImageNet profile 元数据，使 batch 准备和模型构建使用同一个 3 通道契约。
- `configurable-multimodal-fusion`: 允许新增模块化 fusion 模型复用现有模态选择语义，并要求 image profile 与启用模态的 batch 输入一致。
- `original-code-compatibility`: 明确当前默认 image 和包含 image 的 fusion 配置迁移到 RGB/ImageNet；旧单通道 image 路径不再提供运行兼容。

## Impact

- 影响代码：`src/kd_sensing/data/transform_ops/image.py`、dataset/batch 准备、canonical config 解析、`src/kd_sensing/models/image.py`、新增模型/backbone/core/head 模块、`src/kd_sensing/models/fusion/`、`src/kd_sensing/engine/*` 构建路径。
- 影响配置：新增 image profile 字段、ResNet-18 encoder 字段、modular sequence model 配置，以及对应 canonical/example 配置；既有 image 和 fusion 配置默认到 RGB/ImageNet。
- 影响依赖：ResNet-18 可优先使用 `torchvision.models.resnet18`；环境需要在 `kd_mm_beam` 中具备 torchvision，或实现清晰的缺失依赖错误。
- 影响测试：新增 RGB image 预处理、encoder、注册、config validation 和最小训练/forward 相关测试。
- 影响文档：扩展指南需要说明默认 `rgb_imagenet` 路径、固定 image `224x224` 约束和重新训练 checkpoint 的要求。
