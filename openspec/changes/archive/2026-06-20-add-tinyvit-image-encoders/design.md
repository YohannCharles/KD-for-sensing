## Context

当前项目的普通视觉 baseline 按 `model-architecture-extension-contract` 默认走 component baseline：image encoder 通过 `ENCODERS` 注册并由 `modular_sequence` 组合，输入 profile 由 `rgb_imagenet` 约束为 3 通道 224x224，encoder 输出统一为 `[B, T, D_raw]`。现有默认 RGB/ImageNet 路径是 `resnet18_imagenet_rgb`，并且默认配置不能静默回退到旧 CNN。

上游 `wkcn/TinyViT` 提供官方 PyTorch 实现和 Model Zoo，其中 `TinyViT-5M` 与 `TinyViT-11M` 都有 224x224 的 ImageNet-22k distill checkpoint；README 记录 5M 约 5.4M 参数 / 1.3G MACs，11M 约 11M 参数 / 2.0G MACs。上游实现依赖 `timm` 的构建 helper，但本仓库当前运行依赖只有 `torch`、`torchvision` 等，不包含 `timm`。

## Goals / Non-Goals

**Goals:**

- 将 TinyViT-5M 和 TinyViT-11M 作为 opt-in image encoder 接入 `ENCODERS`，覆盖 22k 预训练与 scratch 四个版本。
- 保持现有训练、评估、batch runtime、`ModelOutput` 适配和 config override 方式不变。
- 支持从本地 checkpoint 或上游 URL 加载 22k distill 预训练权重，并记录完整 provenance。
- 提供可审计训练策略 metadata：variant、pretrained、pretrained_source、checkpoint_path/url、freeze policy、trainable stages、backbone_dim、output_dim。
- 不提交上游权重、cache、训练日志、checkpoint 或生成配置。

**Non-Goals:**

- 不把 TinyViT 设为默认 image encoder；默认 canonical image/fusion 配置仍使用 ResNet-18。
- 不引入新的训练 CLI、旧式根脚本、dataset 字段或 image 专用 forward 分支。
- 不实现 TinyViT 的 ImageNet 预训练/蒸馏训练流程；只复用上游发布的 inference checkpoint 作为视觉先验。
- 不接入 TinyViT-21M、384/512 分辨率或其它 foundation vision backbone；本 change 限定 5M/11M 224 版本。

## Decisions

### Decision 1: 作为 `ENCODERS` component baseline，而不是 whole-model

新增四个注册名：

- `tinyvit_5m_scratch_rgb`
- `tinyvit_5m_22k_rgb`
- `tinyvit_11m_scratch_rgb`
- `tinyvit_11m_22k_rgb`

这些注册名都构建同一个窄 encoder 类，只是固定 `variant` 与 `pretrained_source` 默认值。用户通过 `model.primary.encoders.image.type` opt-in，例如把现有 image/fusion 配置 override 到 `tinyvit_5m_22k_rgb`。

Rationale: TinyViT 只替换帧级视觉表征提取，不改变时序建模、fusion、head、loss 或 batch runtime。把它做成 whole-model 会扩大模型表面积并重复 `modular_sequence` 已有能力。

### Decision 2: 本地适配 TinyViT 最小架构，避免新增必需 `timm`

实现放在 `src/kd_sensing/models/tinyvit.py` 或等价窄模块，保留上游 MIT license/copyright 注释，只适配 224 分辨率的 5M/11M 结构。`DropPath`、`trunc_normal_`、state dict filter 和 URL loading 使用 PyTorch/torchvision 已有能力或本地小 helper，避免把 `timm` 变成新的项目必需依赖。

Alternative: 直接新增 `timm` 依赖并复用上游 `_create_tiny_vit`。这样代码更短，但会扩大环境依赖，且构建 helper 对不同 `timm` 版本有分支。首版优先本地适配；若后续确实需要更多 TinyViT family，再单独评估 `timm`。

### Decision 3: 输入严格遵守 `rgb_imagenet`

TinyViT encoder 接收 `[B, T, 3, 224, 224]`，并在构建时调用 `validate_image_encoder_profile`。forward 遇到非 5 维、非 3 通道或非 224x224 时抛出包含实际 shape 的错误；不在 encoder 内隐式 resize。

Rationale: 现有 ResNet-18 路径已经使用严格 224 profile。TinyViT 继续复用这个契约，可以避免不同 encoder 各自定义 resize/normalize 语义，也能保持 cache 与 diagnostics 的输入一致性。

### Decision 4: 特征提取移除分类头并投影到配置维度

TinyViT backbone 输出每帧 pooled feature，5M 的 backbone feature dim 为 320，11M 为 448。encoder 使用 `norm_head(forward_features(frames))` 得到帧级 feature，然后通过可训练 projection 映射到 `output_dim` / `feature_size` / `d_model`。返回值始终是 `[B, T, output_dim]`，不返回 ImageNet logits。

Rationale: 这与 `ResNet18ImageEncoder` 的“去分类头 + projection”契约一致，后续 projector/core 不需要知道 TinyViT 内部通道数。

### Decision 5: 22k 权重加载显式、可审计、可测试

`tinyvit_*_22k_rgb` 默认 `pretrained=True`、`pretrained_source="imagenet22k_distill"`，并使用上游 checkpoint 命名约定加载 22k distill 权重。加载逻辑必须：

- 支持 `checkpoint_path` 优先，用于离线环境和本地缓存。
- 支持 `allow_download=true` 时通过 URL 下载到 torch hub cache。
- 过滤 `attention_bias_idxs`、分类 `head.*` 等不属于 downstream encoder 的权重。
- 对非预期 missing/unexpected keys 或 shape mismatch 抛出清晰错误。
- 在 `training_strategy_metadata()` 中记录最终来源、URL/path、是否下载、过滤策略和 checkpoint schema。

Scratch 注册名默认 `pretrained=False`，不得触发下载。测试中通过 monkeypatch URL loader 或小型 fake state dict 覆盖预训练分支，不能依赖真实网络或下载权重。

### Decision 6: 冻结策略复用 ResNet-18 风格

默认 `freeze_backbone=True`，只训练 projection、后续 projector/core/head。配置可选：

- `freeze_backbone: false` 全量微调。
- `unfreeze_stages` 显式解冻 `patch_embed`、`layer0`、`layer1`、`layer2`、`layer3`、`norm_head`。
- `unfreeze_last_n_stages` 保守解冻最后若干 TinyViT layer。

metadata 记录 `trainable_stages` 和实际 trainable parameter count。普通 baseline 不消费 reliability metadata，TinyViT encoder 不需要新增 batch 字段。

### Decision 7: 配置以 override 和文档示例为主

首版不新增新的 canonical image/fusion YAML，不改变 `docs/maintainer_context_index.yaml` 的 canonical config surface。用户可用现有配置加 override 选择 TinyViT：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/image/supervised.yaml \
  -o model.primary.encoders.image.type=tinyvit_5m_22k_rgb
```

若后续需要固定实验矩阵，再通过单独 change 或已有 experiment recipe 管理，不把大量 TinyViT 实体 YAML 放进 canonical 配置根目录。

## Risks / Trade-offs

- [Risk] 本地适配上游架构可能与官方实现漂移。→ Mitigation: 只实现 5M/11M 224 必需路径，保留上游参数表和 license 注释，用 synthetic forward 与 fake checkpoint load 锁定接口；必要时增加与上游 state dict key 的 fixture。
- [Risk] 22k checkpoint 下载在无网络环境失败。→ Mitigation: 支持 `checkpoint_path`，下载失败给出清晰错误；scratch 版本不触发下载。
- [Risk] 预训练分类头维度 21841 与 downstream encoder 不匹配。→ Mitigation: 明确过滤 `head.*`，只加载 backbone/norm 权重，metadata 记录过滤策略。
- [Risk] TinyViT 训练成本或显存高于预期。→ Mitigation: 默认冻结 backbone，projection 可训练；用户显式选择微调 stage。
- [Risk] 新注册名进入 registry allowlist 后默认导入触发重依赖或下载。→ Mitigation: 模块导入只定义类和 URL 常量，下载只发生在构建 22k encoder 且未提供本地 path 时。

## Migration Plan

1. 新增 TinyViT spec delta 和 focused tests，先锁定四个注册名、输入输出、metadata、权重加载和错误语义。
2. 实现 `TinyViTImageEncoder` 与 5M/11M 架构 helper，注册四个 canonical encoder 名称。
3. 接入默认组件导入和 registry allowlist，确保 `import kd_sensing.registries` 仍轻量。
4. 补充 README/扩展指南中的 opt-in override 示例，不改变默认 image/fusion 配置。
5. 运行 OpenSpec 校验、TinyViT focused tests、配置加载 smoke 和架构边界测试。

Rollback: 删除新增 TinyViT 模块、注册名、tests 和文档示例即可；由于默认配置未切换到 TinyViT，回滚不会影响现有 ResNet-18、JEPA 或 Camera AE 路线。

## Open Questions

- 22k 预训练版本默认是否允许自动下载上游权重，还是要求用户显式提供 `checkpoint_path`？本设计倾向允许显式 opt-in 的 22k 注册名自动下载，但测试和 CI 不依赖网络。
- 是否需要把 TinyViT 纳入当前 active `jepa-visual-architecture-sweep` 的后续实验矩阵？本 change 只提供 encoder 能力，是否跑 sweep 由另一个实验 change 决定。
