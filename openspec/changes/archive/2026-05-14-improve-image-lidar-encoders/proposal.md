## Why

当前单模态对比显示 camera baseline 明显低于论文中的强 camera-only 结果，而 LiDAR baseline 在 Scenario 31 上接近多数类猜测，说明默认 encoder 和 LiDAR 数据路径没有提供可靠的视觉/几何表征。需要把默认 camera encoder 对齐为 ImageNet 预训练 ResNet-18，并把 LiDAR 修复为可诊断、可训练、可回归验证的 baseline。

## What Changes

- 将默认 camera/image-only teacher/student/KD 配置从从头训练的小 CNN 切换为 `rgb_imagenet` + ImageNet 预训练 ResNet-18 encoder，不再发货旧小 CNN image 配置入口。
- 让包含 image 的 canonical fusion 配置默认使用同一 ResNet-18 image encoder，避免 image-only 和 fusion 中的 camera 表征不一致。
- 修复 LiDAR 默认训练路径，使 LiDAR BEV 归一化、cache、ROI/FoV 和增强配置可复用且默认不退化到多数类 baseline。
- 增加 LiDAR 质量诊断与回归验证：至少覆盖 BEV 非空率、通道统计、训练/测试复用 normalizer、LiDAR-only Top-K 指标和多数类/last-beam sanity baseline 对比。
- 更新测试和文档，明确默认 image/LiDAR baseline 统一走 modular encoder profile。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `resnet18-image-encoder`: 默认 camera encoder 行为改为使用 ImageNet 预训练 ResNet-18，并要求 image-only 与包含 image 的 canonical fusion 配置保持一致。
- `lidar-preprocessing`: LiDAR 默认预处理必须支持训练集统计归一化、cache 参数隔离、ROI/FoV 可追踪和质量诊断，避免 BEV 输入静默退化。
- `lidar-modality-model`: LiDAR-only baseline 必须能使用修复后的 BEV 输入训练到高于多数类/退化 baseline 的可验证水平，并暴露必要模型/配置入口。
- `experiment-workflow`: 默认实验配置、日志和验证流程必须记录 encoder/preprocessing 选择，并提供可横向比较的 image/LiDAR 单模态回归检查。

## Impact

- 影响 `configs/image/`、包含 image 的 `configs/fusion/` canonical 配置、配置生成/解析逻辑和相关 tests。
- 影响 `src/kd_sensing/models/image_encoders.py`、modular image encoder 注册和 image-only 模型构建路径。
- 影响 `src/kd_sensing/data/transform_ops/lidar.py`、LiDAR dataset normalizer/cache 使用、LiDAR-only 模型配置和训练输出 metadata。
- 可能增加对 `torchvision` 预训练权重可用性的要求；缺少依赖时必须给出清晰错误。
- 不在默认或 canonical 配置中保留旧 image/LiDAR encoder 入口；旧注册类如仍存在，只作为内部兼容实现，不作为配置入口发布。
