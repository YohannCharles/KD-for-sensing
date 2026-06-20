## Why

当前 image/fusion 主线已经有 `rgb_imagenet` + ResNet-18 作为默认视觉 encoder，但视觉 backbone 对照仍缺少小型 Transformer / CNN-Transformer hybrid 的可配置选项。TinyViT-5M 和 TinyViT-11M 提供更轻量的 ImageNet-22k distill 预训练视觉先验，适合作为可选 image encoder 与 ResNet-18、JEPA visual encoder 和 Camera AE 路线做统一协议比较。

## What Changes

- 新增 TinyViT image encoder capability，作为 `ENCODERS` 组件接入 `modular_sequence`，不新增训练脚本、不复制训练循环、不替换现有默认 ResNet-18。
- 支持 4 个 opt-in 版本：`TinyViT-5M` 未预训练、`TinyViT-5M` ImageNet-22k distill 预训练、`TinyViT-11M` 未预训练、`TinyViT-11M` ImageNet-22k distill 预训练。
- TinyViT encoder 接收当前 `rgb_imagenet` profile 的 `[B, T, 3, 224, 224]` image batch，输出 `[B, T, D]` 帧级 embedding，并通过投影层适配现有 projector/core/head。
- 预训练权重只通过显式配置启用，支持本地 checkpoint 路径和可审计的上游 URL/缓存策略；源码不提交权重、cache、checkpoint 或运行产物。
- 训练策略支持默认冻结 backbone、只训练投影层、解冻最后若干 stage 或全量微调，并在 metadata 中记录 variant、预训练来源、checkpoint policy、freeze/unfreeze 策略和输出维度。
- 新增最小配置样例和 focused tests，覆盖 registry 构建、synthetic forward、profile/shape 错误、预训练权重加载分支、metadata 和配置加载。
- **BREAKING**: 无。现有 image/fusion 默认配置继续使用 `resnet18_imagenet_rgb`，TinyViT 仅作为显式 opt-in encoder。

## Capabilities

### New Capabilities

- `tinyvit-image-encoder`: 定义 TinyViT-5M/11M 作为可选 RGB/ImageNet image encoder 的注册名、预训练/未预训练版本、输入输出、权重加载、训练策略 metadata 和验证契约。

### Modified Capabilities

- `component-registry`: 默认组件导入和注册发现需包含新增 TinyViT encoder 名称，同时保持 `kd_sensing.registries` 轻量导入，不 eager import 权重或触发下载。

## Impact

- 影响代码区域：`src/kd_sensing/models/` 新增 TinyViT 架构/encoder 窄模块，`src/kd_sensing/registries.py` 默认组件导入需注册该模块，必要时更新 image profile 校验 helper。
- 影响配置：新增 opt-in image-only 与 image+GPS/fusion 样例配置，或新增可复用 overlay/recipe；默认 canonical `configs/image/{strong,lightweight,supervised}.yaml` 不改变。
- 影响依赖：优先采用本地适配实现，避免新增运行时必需依赖；若实现选择复用 `timm`，必须在 design/tasks 中明确依赖和缺失依赖错误。
- 影响测试：新增 TinyViT focused tests，并运行架构边界、配置加载和相关 registry/model forward smoke。
- 影响文档：README 或扩展指南只需补充 TinyViT 是 opt-in image encoder；OpenSpec 记录完整行为契约。
