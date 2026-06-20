## 1. 契约测试

- [x] 1.1 新增 TinyViT focused test 文件，覆盖四个 `ENCODERS` 注册名：`tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb`、`tinyvit_11m_22k_rgb`。
- [x] 1.2 添加 synthetic forward 测试，验证 TinyViT encoder 接收 `[B, T, 3, 224, 224]` 并输出 `[B, T, D]`，且不返回 ImageNet logits 或 beam logits。
- [x] 1.3 添加 profile/shape 错误测试，覆盖非 `rgb_imagenet`、非 3 通道、非 224x224 和非 5 维输入的清晰错误信息。
- [x] 1.4 添加 22k 权重加载测试，通过 monkeypatch/fake state dict 覆盖 `checkpoint_path`、URL loader、分类 head/filter、unexpected key 和 scratch 不下载分支，测试不得依赖真实网络。
- [x] 1.5 添加 freeze/unfreeze metadata 测试，覆盖默认冻结、`freeze_backbone=false`、`unfreeze_stages`、`unfreeze_last_n_stages` 和未知 stage 拒绝。
- [x] 1.6 添加 `modular_sequence` 集成 smoke，验证 image-only 和 image+GPS 配置可用 TinyViT encoder 构建并聚合 training strategy metadata。

## 2. TinyViT encoder 实现

- [x] 2.1 新增 `src/kd_sensing/models/tinyvit.py` 或等价窄模块，保留上游 TinyViT/Microsoft MIT license/copyright 说明，并只实现 224 分辨率的 TinyViT-5M/11M 必要结构。
- [x] 2.2 用项目现有依赖实现 TinyViT 所需的 DropPath、truncated normal init、patch embed、MBConv、window attention、patch merging、basic layers 和 feature extraction，避免新增必需 `timm` 依赖。
- [x] 2.3 实现 `TinyViTImageEncoder`，严格校验 `[B, T, 3, 224, 224]` 输入，将 TinyViT pooled feature 经 `norm_head` 和 projection 输出 `[B, T, output_dim]`。
- [x] 2.4 实现 5M/11M variant 参数表，确保 5M backbone_dim 为 320、11M backbone_dim 为 448，并用 `_resolve_output_dim` 兼容 `output_dim`、`feature_size` 和 `d_model`。
- [x] 2.5 实现 22k checkpoint loader，支持本地 `checkpoint_path` 优先、上游 URL/torch hub cache、`model` payload schema、`attention_bias_idxs` 与 `head.*` 过滤、错误诊断和 metadata provenance。
- [x] 2.6 实现 TinyViT freeze/unfreeze 策略和 `training_strategy_metadata()`，记录 variant、pretrained、pretrained_source、checkpoint source、freeze_backbone、trainable_stages、backbone_dim、output_dim 和 reliability metadata 消费状态。

## 3. Registry 与配置表面积

- [x] 3.1 在 TinyViT 模块中注册四个 canonical `ENCODERS` 名称，不注册 legacy alias，不新增 whole-model `MODELS` 入口，除非实现中需要与现有 ResNet encoder 对称暴露并在代码注释中说明。
- [x] 3.2 更新 `import_default_components()`，确保默认组件导入会注册 TinyViT encoder，同时 `import kd_sensing.registries` 仍不导入 TinyViT 权重、不触发下载、不要求 `timm`。
- [x] 3.3 更新架构边界或 registry allowlist，使四个 TinyViT encoder 名称被视为当前 canonical opt-in 组件，并保持退役 alias guard 不变。
- [x] 3.4 如需修改 `validate_image_encoder_profile`，只扩展 TinyViT 的 `rgb_imagenet` 校验，不放宽 ResNet-18 和旧 image encoder 的 migration guard。
- [x] 3.5 不新增 canonical image/fusion YAML；如需示例，优先在 README/扩展指南提供 `-o model.primary.encoders.image.type=...` override。

## 4. 文档与使用说明

- [x] 4.1 在 README 或模型扩展指南补充 TinyViT opt-in override 示例，说明默认配置仍使用 ResNet-18。
- [x] 4.2 文档记录四个版本的含义、输入 profile、22k checkpoint 来源、本地 checkpoint 优先策略、scratch 版本不下载权重和默认冻结策略。
- [x] 4.3 文档明确 TinyViT 只作为 image encoder 组件，不恢复 KD/distillation 训练流程，不提交权重、cache、checkpoint 或训练输出。

## 5. 验证

- [x] 5.1 运行 `openspec validate add-tinyvit-image-encoders --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_tinyvit_image_encoder.py -q` 或实现后的对应 TinyViT focused test。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`，确认现有 canonical 配置默认仍是 ResNet-18 且 TinyViT override 可解析。
- [ ] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认 registry allowlist、轻量导入和产物边界没有回归。
- [x] 5.5 如实现触碰 modular sequence metadata，运行相关模型 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py -q`。
