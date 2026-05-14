## Context

当前 image 数据路径统一为 RGB/ImageNet：dataset 直接加载 RGB 帧，batch 准备输出 `[B, T, 3, 224, 224]`，模型侧需要兼容三通道视觉输入。仓库曾经引入过 M2BeamLLM 风格 encoder，后来因为效果和维护成本退役，因此这次新增 ResNet-18 ImageNet 路径时不恢复退役注册名，也不恢复已删除的旧单通道 image 分支。

现有模型已经有多种形态：单模态是 `feature extractor -> LayerNorm -> GRU -> attention/head`，fusion 是各模态 extractor 后 concat/projection，再进入 GRU/attention/head，CRAF/MARF 则在 token core 中同时处理模态关系和时间关系。新增架构应把“融合 + 预测”的组合抽象成 `representation_core`，避免强制拆成固定的 Fusion/Predictor 两段。

## Goals / Non-Goals

**Goals:**

- 将 `rgb_imagenet` image profile 作为唯一 image 输入契约，输出 3 通道 224x224 标准化 RGB 帧。
- 新增 ResNet-18 image encoder，支持 ImageNet 预训练、冻结/解冻策略和统一 `[B, T, D]` 输出。
- 新增模块化序列模型入口：`encoders -> projectors -> representation_core -> heads`，其中 `representation_core` 统一承载时序建模和跨模态建模。
- 建立 image profile、encoder 输入通道和模型配置的早期校验，避免训练中途才出现 conv shape mismatch。
- 提供 ResNet-18 image-only baseline 和可扩展到 fusion 的配置骨架。

**Non-Goals:**

- 不恢复已退役的 `m2beamllm_*` 注册名或 `encoder_profile: m2beamllm`。
- 不恢复已删除的旧单通道 image 预处理、cache、encoder 或 checkpoint 兼容。
- 不在本次变更中重写 CRAF/MARF 主体；只提供可复用的模块化 core 接入边界。
- 不新增 Scenario 33/34 场景注册，除非后续单独提出场景支持变更。
- 不把 auxiliary blockage/position loss 直接塞进 head；head 只产出，loss 仍由训练侧配置控制。

## Decisions

1. image preprocessing 固定为 RGB/ImageNet。

   配置字段 `data.dataset.image_profile` 标准化为 `rgb_imagenet`。dataset 调用 `build_rgb_imagenet_transform()` 和 `load_rgb_imagenet_frames()` 或等价函数：

   ```text
   rgb_imagenet:
     RGB paths -> RGB -> resize/crop 224x224 -> float [0,1] -> ImageNet normalize
     dataset image: [seq_len, 3, 224, 224]
     model input after batch prep: [B, seq_len + num_pred - 1, 3, 224, 224]
   ```

   run metadata 记录实际 `image_profile`、输入通道和 processed image 来源。诊断 manifest 的 processed image 表示来自 dataset 返回张量的反标准化预览。

2. ResNet-18 作为新 encoder，不直接改旧模型注册名。

   新增 `ResNet18ImageEncoder`，位置建议为 `src/kd_sensing/models/image_encoders.py`。它负责：

   - 接收 `[B, T, 3, 224, 224]`。
   - reshape 为 `[B*T, 3, 224, 224]` 调用 torchvision ResNet-18。
   - 去掉 `fc` 分类层，得到 512 维 pooled feature。
   - 通过 projection 输出配置指定的 `feature_size` 或 `d_model`。
   - reshape 回 `[B, T, D]`。

   预训练接口优先使用当前 torchvision 推荐的 weights API；缺少 torchvision 时在构建阶段报错。默认训练策略建议 `freeze_backbone: true` 或只解冻 `layer4`，先稳定验证 RGB 路径，再考虑全量微调。

3. 新增 `ModularSequenceModel`，把 Fusion/Predictor 合并为 `representation_core`。

   目标结构：

   ```text
   ModularSequenceModel
     encoders:
       image: resnet18_imagenet_rgb
       radar: radar_cnn
       gps: gps_mlp
       lidar: lidar_cnn
       mmwave: mmwave_mlp
     projectors:
       per modality D_raw -> d_model
     representation_core:
       single_gru | early_concat_gru | token_transformer | craf_core | marf_core
     heads:
       beam_head
       optional blockage_head / position_head
   ```

   encoder 只输出 `[B, T, D_raw]`，projector 统一到 `[B, T, d_model]`。单模态 core 接收 `[B, T, d_model]`；多模态 core 接收 `[B, K, T, d_model]` 或等价 token 结构。beam head 输出 `[B, T, num_classes]`。为了兼容现有训练/KD/eval，forward 返回仍应能被 `ModelOutput` 适配为 logits、input_features、output_features；dict 输出可以额外带 diagnostics 和 auxiliary heads。

4. image profile 校验放到配置解析和模型构建边界。

   推荐新增中心化校验函数，例如 `resolve_image_profile()` 和 `validate_image_encoder_profile()`，并在以下路径调用：

   - `config/io.py` 或 canonical config 解析：标准化默认值，校验 `image_size` 和 profile。
   - dataset 构建：只选择 RGB/ImageNet loader。
   - model 构建：校验 encoder 支持的 `image_channels` 与 profile 通道一致。
   - batch 准备：形成 `[B, T, 3, 224, 224]` 并使用统一 future padding 规则。

5. canonical 配置采用新增入口，旧 checkpoint 默认路径不再使用。

   新增配置建议：

   - `configs/image/resnet18_teacher_no_kd.yaml`
   - `configs/fusion/image_gps_resnet18_modular_no_kd.yaml`
   - 后续可补充 `configs/fusion/image_radar_resnet18_modular_*.yaml`

   既有 image 和包含 image 的 fusion 配置解析后也应得到 `image_profile: rgb_imagenet`，且不应默认指向旧外部 image 权重。KD 配置应优先通过当前 registry 找到重新训练后的 teacher checkpoint，缺失时要求用户提供当前 RGB 路径权重。

## Risks / Trade-offs

- RGB/ImageNet 输入显存和 I/O 成本高于旧单通道输入 → 先支持冻结 ResNet-18、较小 batch、AMP，并用 profile 工具量化 DataLoader/transfer/model step。
- ImageNet 预训练存在 domain shift → 默认不全量微调，提供冻结/解冻策略和新配置。
- torchvision 版本 API 可能变化 → 只在 ResNet-18 encoder 构建时延迟导入，并对缺失依赖或 weights API 不可用给出明确错误。
- 模块化模型抽象过宽可能延误交付 → 第一阶段只实现 `resnet18_imagenet_rgb + single_gru/early_concat_gru + beam_head` 的最小闭环，CRAF/MARF core adapter 后续迭代接入。
- checkpoint 不兼容风险 → 保持严格加载；新增 ResNet-18 注册名和配置，不复用旧 checkpoint 默认路径。

## Migration Plan

1. 增加 image profile 常量、标准化和校验函数；更新默认配置，使未声明 profile 的默认路径解析为 `rgb_imagenet`。
2. 在 image transform 模块新增 RGB/ImageNet transform 与 frame sequence loader；更新 dataset 只加载 RGB/ImageNet image。
3. 更新 batch 准备函数，输出统一的 RGB/ImageNet 模型输入时间长度。
4. 新增 ResNet-18 image encoder。
5. 新增 `ModularSequenceModel`、基础 projector、`single_gru`/`early_concat_gru` core 和 beam head。
6. 接入注册表和默认组件导入，新增 ResNet-18 image-only 配置。
7. 补充单元测试和 smoke forward 测试；使用 `conda run -n kd_mm_beam pytest ...` 运行最小回归。
8. 更新 README 或扩展指南，说明 RGB/ImageNet image profile、固定尺寸和 checkpoint 重新训练要求。

回滚策略：删除新增 ResNet-18/modular 配置和模块，并恢复到上一版本；本变更不提供旧单通道 image 路径的运行时回退。

## Open Questions

- 是否在第一阶段提供 ResNet-18 student，还是先提供 ResNet-18 teacher/baseline。
- Scenario 31-34 的场景注册是否与本变更一起做。当前请求聚焦 image preprocessing 和 encoder，场景 33/34 支持更适合单独变更。
- RGB/ImageNet 是否需要离线 transformed-frame cache。第一阶段建议不做，先避免引入第二套大体积 image cache。
