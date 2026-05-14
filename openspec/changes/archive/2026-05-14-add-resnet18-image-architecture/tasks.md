## 1. 配置与契约

- [x] 1.1 在中心化模态契约中新增 image profile 元数据，覆盖 `rgb_imagenet` 的通道数、默认尺寸、cache 能力和推荐 encoder。
- [x] 1.2 在默认配置和配置解析中新增 `data.dataset.image_profile`，未配置时标准化为 `rgb_imagenet`。
- [x] 1.3 增加 `resolve_image_profile()` 和 image profile 校验逻辑，拒绝未知或已删除 profile，并让错误信息列出可用值。
- [x] 1.4 更新现有 image size 校验，为 `rgb_imagenet` 明确 ResNet-18 的 224x224 输入约束。
- [x] 1.5 在 run metadata 中记录实际使用的 `image_profile`、image 输入通道和 ResNet-18 训练策略。

## 2. Image 预处理与 batch 准备

- [x] 2.1 新增 RGB/ImageNet transform 和 frame sequence loader，输出 `[seq_len, 3, 224, 224]` float tensor，并使用 ImageNet mean/std 标准化。
- [x] 2.2 更新 `DeepSense6GDataset`，让 image modality 只使用 RGB/ImageNet loader。
- [x] 2.3 更新 image batch 准备逻辑，形成模型侧 `[B, T, 3, 224, 224]` 输入，并保持 future padding 语义一致。
- [x] 2.4 更新诊断 manifest 或 viewer metadata，使 processed image 来源记录为 RGB/ImageNet profile。

## 3. ResNet-18 Encoder

- [x] 3.1 新增 `ResNet18ImageEncoder`，延迟导入 torchvision，支持 ImageNet 预训练权重、移除分类层和 `[B, T, 3, 224, 224] -> [B, T, D]` 输出。
- [x] 3.2 为 ResNet-18 encoder 实现 projection、dropout 和可配置输出维度，保证输出能对齐现有 `feature_size` 或模块化 `d_model`。
- [x] 3.3 实现冻结 backbone、只训练投影层、选择性解冻 stage 和全量微调的参数控制。
- [x] 3.4 增加 encoder/profile 匹配校验，ResNet-18 必须绑定 `rgb_imagenet` 和 3 通道输入，已删除 encoder 名称必须被拒绝。

## 4. 模块化序列模型

- [x] 4.1 新增 projector 组件，将每个模态 encoder 输出映射到统一 `d_model`。
- [x] 4.2 新增基础 representation core：至少实现 `single_gru` 和 `early_concat_gru`，并预留 token transformer/CRAF/MARF core adapter 接口。
- [x] 4.3 新增 beam classification head，输出 `[B, T, num_classes]` logits，并保留辅助 head 扩展点。
- [x] 4.4 新增 `ModularSequenceModel` 注册入口，按 `modalities` 构建 encoders、projectors、representation core 和 heads。
- [x] 4.5 确保模块化模型 forward 输出兼容现有 `ModelOutput`、loss、metric、KD 和 evaluator 路径。
- [x] 4.6 为多模态模块化模型增加 batch/time 维一致性检查，错误信息包含模态名和实际 shape。

## 5. 注册与配置

- [x] 5.1 将 ResNet-18 image encoder、模块化模型、core 和 head 接入现有注册/默认组件导入边界。
- [x] 5.2 确保 `import kd_sensing.registries` 不 eager import torchvision、dataset 或训练模块。
- [x] 5.3 新增 ResNet-18 image-only canonical 或 example 配置，显式设置 `image_profile: rgb_imagenet` 和 ResNet-18 encoder。
- [x] 5.4 为模块化 fusion 增加最小 example 配置，复用现有 `modalities` 语义，并演示 image+radar 或 image+gps 输入。
- [x] 5.5 确保 `configs/image/*.yaml`、包含 image 的 fusion 配置和 `image_radar_*` canonical 配置解析后使用 RGB/ImageNet 路径。

## 6. 测试与文档

- [x] 6.1 新增 image profile 单元测试，覆盖默认 profile、未知 profile、已删除 profile 拒绝和 RGB/ImageNet shape/标准化。
- [x] 6.2 新增 ResNet-18 encoder 测试，覆盖输出 shape、冻结策略、缺少 torchvision 的清晰错误和 encoder/profile 不匹配错误。
- [x] 6.3 新增模块化模型 forward 测试，覆盖 image-only ResNet-18、image+GPS 或 image+radar fusion 的最小 batch。
- [x] 6.4 新增配置兼容测试，确认默认 profile 为 `rgb_imagenet`，且 ResNet-18 配置不会复用旧 checkpoint。
- [x] 6.5 更新 README 或扩展指南，说明默认 `rgb_imagenet + resnet18_imagenet_rgb`、固定 image 224x224 约束和 checkpoint 重新训练要求。
- [x] 6.6 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_fusion_image_feature_extractor.py` 运行现有 image/fusion 回归。
- [x] 6.7 使用 `conda run -n kd_mm_beam pytest` 运行完整测试套件；如耗时过长，记录未运行的测试范围和原因。
