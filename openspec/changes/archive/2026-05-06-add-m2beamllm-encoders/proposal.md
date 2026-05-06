## Why

当前 image、radar、GPS、LiDAR 的 GRU 前编码器是项目内自定义实现，和 M2BeamLLM 论文中的多模态 sensing data encoding 不一致，难以直接做论文处理方式的对照实验。需要新增一套可选编码路径，只替换 GRU 之前的 encoding 部分，保持 GRU 及其后的预测头、KD 训练流程和既有配置兼容。

## What Changes

- 新增 M2BeamLLM 风格的非 mmWave 四模态编码器：
  - image：输入 resize/reshape 到 224x224，按 ImageNet mean/std 标准化，使用去掉分类头的 ResNet-18，再经过 FC/ReLU 投影到 `feature_size`。
  - radar：支持 M2BeamLLM 风格的 Range/Angle FFT 后 RA map 编码路径，并通过 CNN/MLP 投影到 `feature_size`；若配置继续使用现有预处理 RA/DA map，则必须明确适配输入契约。
  - LiDAR：支持点云转单通道 256x256 histogram，点计数裁剪到 5 并归一化到 [0, 1]，再用改造 ResNet-18 投影到 `feature_size`。
  - GPS：使用经训练集统计得到的 min-max 归一化坐标，再通过三层 MLP + LayerNorm + GELU 投影到 `feature_size`。
- 新增可选模型/编码器注册项或配置 profile，优先采用新增名称，不覆盖当前 `image_teacher`、`radar_teacher`、`gps_teacher`、`lidar_teacher`、`fusion_teacher`、`*_student` 的默认行为。
- 保持 GRU 及其之后的时序建模、attention/classifier、输出 `(pred, features, enhanced_seq_out)` 契约和 KD 损失接口不变。
- mmWave 模态保持现状，不引入 M2BeamLLM encoder 替换。
- 为单模态和 fusion 提供最小可运行配置示例，允许用户选择 M2BeamLLM encoder 进行新实验。

## Capabilities

### New Capabilities
- `m2beamllm-modality-encoders`: 定义 image、radar、GPS、LiDAR 的 M2BeamLLM 风格 GRU 前编码器、配置启用方式、输入契约、输出契约和 mmWave 排除规则。

### Modified Capabilities

## Impact

- 影响模型代码：`src/kd_sensing/models/image.py`、`src/kd_sensing/models/radar.py`、`src/kd_sensing/models/gps.py`、`src/kd_sensing/models/lidar.py`、`src/kd_sensing/models/fusion/networks.py`，以及模型注册导出。
- 影响数据/预处理：可能需要扩展 image 标准化、LiDAR histogram、GPS min-max artifact、radar raw/RA 输入适配，但不得破坏现有 dataset 字段和缓存策略。
- 影响配置：新增单模态和 fusion 的 M2BeamLLM encoder 配置或 profile；默认 canonical 配置保持不变。
- 影响依赖：若当前环境未稳定提供 torchvision ResNet-18，需要在项目依赖和测试中显式处理。
- 影响测试：新增 encoder shape、注册、配置构建、mmWave 排除、GRU 后结构不变的回归测试。
