## Why

当前项目已经支持 image、radar、GPS 及其可选融合，但还不能读取 DeepSense LiDAR 点云、生成可训练的 LiDAR 表征，也不能通过统一训练入口运行 LiDAR-only 或 LiDAR 参与的 fusion 实验。引入 LiDAR 模态可以补齐场景 31-34 多模态波束预测所需的空间几何信息，并为 BEV/FoV 裁剪等消融实验提供统一代码路径。

## What Changes

- 新增 LiDAR 点云预处理能力：读取原始点云路径，做无效点过滤、ROI/FoV 裁剪、可选地面/背景过滤、BEV 伪图像化和训练集统计归一化。
- 扩展序列 CSV 生成和 Scenario9 dataset，使启用 LiDAR 时样本包含 `lidar` 字段，形状与 image/radar 风格一致，为 `[seq_len, channels, height, width]`。
- 新增 LiDAR 模型族：`LidarFeatureExtractor`、`LidarModalityNet` 和 `LidarStudentModalityNet`，保持 `(pred, features, output_features)` 输出契约和 teacher/student 命名风格。
- 扩展训练、验证、评估和 batch 准备流程，支持 `experiment.task: lidar`，并支持 LiDAR 在 fusion `modalities` 中被选择。
- 新增 LiDAR-only no-KD、student no-KD、logits KD、RKD 配置，以及包含 LiDAR 的 fusion 配置示例。
- 新增测试覆盖 LiDAR BEV 构造、dataset 字段、模型 forward、batch 准备、配置构建和 fusion 模态校验。

## Capabilities

### New Capabilities
- `lidar-preprocessing`: 定义 LiDAR 序列列生成、点云读取、ROI/FoV 裁剪、BEV 伪图像化、归一化和 dataset batch 字段契约。
- `lidar-modality-model`: 定义 LiDAR-only teacher/student 模型、输入输出契约、KD 兼容性和默认实验配置。

### Modified Capabilities
- `configurable-multimodal-fusion`: 将 `lidar` 纳入 fusion 可选模态，要求 teacher/student fusion 分支和输入校验支持 LiDAR。
- `experiment-workflow`: 将 `lidar` 纳入配置驱动训练、评估、预处理和默认配置检查。
- `component-registry`: 要求 LiDAR 模型和预处理器可通过现有注册机制发现与构建。

## Impact

- 影响代码：`src/kd_sensing/preprocessing/`、`src/kd_sensing/data/`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/builders.py`、`src/kd_sensing/models/`、`src/kd_sensing/models/fusion/`、`src/kd_sensing/config/defaults.py`、`configs/`、`tests/`。
- 数据接口：序列 CSV 需要可选 `lidar1..lidarN` 列；启用 LiDAR 时 dataset 需要从这些路径加载 BEV 张量。
- 运行接口：新增 `experiment.task: lidar` 和 fusion `modalities: [..., lidar]`；旧 image/radar/GPS 配置必须保持兼容。
- 依赖影响：优先使用 NumPy 解析 ASCII PCD 或文本点云，避免强制引入 Open3D；若后续需要二进制 PCD 支持，可再评估可选依赖。
