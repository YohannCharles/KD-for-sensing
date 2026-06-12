## Context

仓库当前已经具备 DeepSense6G 场景化数据构建、模态感知加载、模型 registry、canonical/virtual config、训练评估 engine、BeamBench 指标和 Camera AE/ResNet/GPS/Transformer fusion 的部分实现积木。用户希望得到一套面向代码模型搜索与实验设计的 baseline 方案，核心任务是用 5 帧 RGB 图像序列和 GPS 序列预测 64 类最优 mmWave beam。

这次变更应保持项目当前边界：训练和评估继续通过 `kd-sensing-train`、`kd-sensing-evaluate` 或包内 CLI 运行；新增模型进入 `src/kd_sensing/models` 及 registry；配置进入 canonical recipe 或实体 YAML；本地数据、日志、cache、checkpoint 和真实训练结果不进入源码变更。

## Goals / Non-Goals

**Goals:**

- 提供四类可并排比较的 baseline preset：Camera AE + GPS、ResNet + GPS late fusion、Transformer token fusion、GPS-only neural baseline。
- 保持所有 preset 输出统一的 64-class logits，并能通过现有 supervised loss、checkpoint、TensorBoard 和 top-k 指标流程训练评估。
- 复用现有 image/GPS 数据字段、GPS 归一化、image profile、DataLoader 参数和 scene selection 语义。
- 为真实数据不可用时的实现验证提供 mock 或小 batch smoke 测试，不伪造真实指标。
- 让 Camera AE + GPS Direct 与已有 BeamBench 复现规格对齐，明确其是套件中的 paper-style preset。

**Non-Goals:**

- 不恢复 KD/distillation、HiST-Beam、residual correction、Top8 selector 或其它已退役路线。
- 不新增一套独立于 `kd_sensing` 包结构的训练脚本工程。
- 不实现官方 DeepSense leaderboard 提交流程或承诺复现官方 Table III 数值。
- 不要求提交真实 DeepSense6G 数据、新训练 checkpoint、TensorBoard 文件或运行输出。

## Decisions

### Decision 1: baseline suite 采用 registry preset，而不是独立脚本

新增模型通过 `MODELS.register()` 暴露，例如 `vision_position_late_fusion`、`vision_position_transformer_fusion` 和必要的 `gps_sequence_baseline`；现有 `camera_ae_frozen`、`resnet18_imagenet_rgb`、`gps_strong/gps_lightweight`、`cls_token_transformer_fusion` 优先复用。配置只指定 `model.primary.type`、encoder 选择、modalities、sequence 聚合和训练参数。

理由：这能复用当前 engine 的 batch 准备、checkpoint、metrics 和产物边界，也符合项目禁止新增旧入口的架构约束。备选方案是生成独立 PyTorch 项目或脚本，但会绕过现有配置、数据契约和健康检查。

### Decision 2: Camera AE + GPS 作为 paper-style preset

Camera AE + GPS baseline 使用 `CameraAEImageEncoder` 读取 AE latent，GPS 使用 direct MLP embedding，二者按时间对齐后 late concat，再用可配置 temporal aggregation 产生每个 horizon 的 logits。默认允许冻结 AE encoder；若配置未提供 checkpoint 且声明 `require_checkpoint=true`，构建必须失败并给出清晰修复提示。

理由：现有 BeamBench 规格要求该路线不得被 residual/gated/attention 模型替代。将其做成受控 preset，可同时服务论文复现和 baseline suite 对照。备选方案是在线训练完整 AE + classifier，但会扩大训练流程复杂度，且不适合作为默认 smoke 路径。

### Decision 3: ResNet + GPS late fusion 使用可替换 image encoder

ResNet baseline 默认使用 `resnet18_imagenet_rgb` encoder，支持冻结 backbone、解冻指定 stage 和输出维度投影；GPS 使用 MLP 或现有 GPS feature extractor；序列聚合默认支持 `mean`，可扩展为 `gru`/`lstm`，但配置必须显式记录聚合方式。

理由：ResNet + GPS 是经典 CV+位置 late fusion，对实验搜索很有用，同时 `ResNet18ImageEncoder` 已有 profile 和训练策略 metadata。备选方案是把 ResNet 写死在 fusion 模型中，会降低后续替换 ViT 或 AE encoder 的能力。

### Decision 4: Transformer 强融合优先复用 CLS-token fusion

Transformer baseline 默认使用 `CLSTokenTransformerFusionNet` 的 image+gps modalities，或在需要更细粒度视觉 token 时新增 narrow wrapper，将 image encoder 输出和 GPS embedding 组织为 token 后进入 `TransformerEncoder`。配置必须记录 token 组织方式、`d_model`、heads、layers、max sequence length 和 positional/type embedding 设置。

理由：仓库已将 lightweight fusion 推荐到 CLS-token transformer，复用它可以减少模型栈分叉。备选方案是引入新的 TransFuser 风格大模型，但在无 LiDAR/radar 的图像+GPS任务中会过度复杂。

### Decision 5: GPS-only neural baseline 与非神经 GPS window baseline 分开

本 change 新增或确认 `gps_sequence_baseline` 仅使用 GPS 序列和合法 split metadata，采用 MLP/GRU/LSTM 预测 beam logits；它不替代已有 `gps-window-baseline-beam-prediction` 的非神经几何 baseline。运行 metadata 必须记录其 `uses_neural_network=true` 和启用字段。

理由：用户明确需要 LSTM/MLP GPS-only 下界，但仓库已有非神经 GPS window baseline，两者实验含义不同，不能混用。

## Risks / Trade-offs

- [Risk] Image profile 与 encoder 输入尺寸不匹配，尤其 ResNet 需要 `[B, T, 3, 224, 224]` 而 Camera AE 可接受 64 尺寸。→ Mitigation：配置必须声明 `image_profile` 和 image size；模型 forward 对 shape 给出清晰错误；配置加载测试覆盖 AE 与 ResNet preset。
- [Risk] 多个 baseline 共享 fusion 配置后，指标字段和报告名混乱。→ Mitigation：run metadata 记录 `baseline_preset`、`encoder_type`、`gps_feature_mode`、`temporal_aggregation` 和 `metric_profile`。
- [Risk] 预训练 ResNet 权重下载或 torchvision 环境不可用导致 smoke 不稳定。→ Mitigation：测试默认使用 `pretrained=false` 或 mock encoder；真实配置可启用 `weights=DEFAULT`，失败时错误指向 `kd_mm_beam` 环境。
- [Risk] Camera AE checkpoint 缺失时用户误以为 baseline 已完整复现。→ Mitigation：`require_checkpoint` 默认在 paper-style preset 中开启；mock/smoke preset 必须标记 `mock_data` 或 `require_checkpoint=false`，报告不得宣称真实结果。
- [Risk] GPS-only neural baseline 可能读取未来 label 或 target oracle 字段。→ Mitigation：数据准备和 metadata 测试要求仅启用 GPS、历史输入和目标 label；预测输入字段在 run metadata 中可审计。

## Migration Plan

1. 增加或整理 vision-position baseline 模型类与 registry 名称，优先组合现有 encoder 与 fusion building blocks。
2. 增加 canonical/virtual config preset：Camera AE + GPS、ResNet + GPS、Transformer image+gps、GPS-only neural。
3. 增加配置加载、模型 forward、mock/small batch training/eval、top-k metrics 和 metadata 测试。
4. 更新 README 或实验矩阵文档，给出 `conda run -n kd_mm_beam ...` 命令和产物位置。
5. 若出现回归，移除新增 preset/config 即可回滚，不需要迁移已有 checkpoint 或数据。

## Open Questions

- Camera AE + GPS paper-style preset 的默认 AE checkpoint 路径是否应指向现有 `All_models/` 资料，还是保持必须由用户显式提供。
- Transformer baseline 是否需要第二阶段支持 ViT patch token；本次默认先使用帧级 visual token，以免引入新重依赖和更高训练成本。
