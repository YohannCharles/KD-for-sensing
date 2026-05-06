## Why

M2BeamLLM 风格的模态编码器实验效果不佳，继续保留会增加模型注册、配置矩阵、数据预处理分支、测试和文档维护成本。现在应删除该可选编码路径，让项目重新聚焦在效果更稳定的现有 image、radar、GPS、LiDAR、mmWave 与 fusion 实现上。

## What Changes

- **BREAKING**：删除 `m2beamllm_*` 单模态 teacher/student 注册名，不再支持通过配置构建 M2BeamLLM 风格 image、radar、GPS、LiDAR 模型。
- **BREAKING**：删除 fusion 中的 `encoder_profile: m2beamllm` 分支，fusion 只保留现有默认 feature extractor 路径。
- 删除 `src/kd_sensing/models/m2beamllm_encoders.py` 及对它的 import、类继承、构造参数和测试依赖。
- 删除 `configs/m2beamllm/` 示例配置与 README 中 M2BeamLLM encoder 对照说明。
- 清理仅服务于 M2BeamLLM encoder 的测试用例；如存在只被该实验路径使用的数据预处理选项或 scaler 分支，也应同步删除或退役。
- 保持现有默认 canonical 配置、非 M2BeamLLM 注册名、训练/KD/eval 流程和 mmWave 分支行为不变。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `m2beamllm-modality-encoders`: 退役 M2BeamLLM 风格模态编码器 capability，要求系统不再提供对应模型注册名、fusion profile、示例配置和专用测试。

## Impact

- 影响代码：`src/kd_sensing/models/m2beamllm_encoders.py`、`src/kd_sensing/models/image.py`、`src/kd_sensing/models/radar.py`、`src/kd_sensing/models/gps.py`、`src/kd_sensing/models/lidar.py`、`src/kd_sensing/models/fusion/networks.py`。
- 影响配置：删除 `configs/m2beamllm/` 下所有实验配置；任何引用 `m2beamllm_*` 或 `encoder_profile: m2beamllm` 的外部配置需要迁移回标准注册名。
- 影响测试：删除或改写 `tests/test_m2beamllm_encoders.py`，并补充回归检查以确认默认模型仍可构建。
- 影响文档：删除 README 中 M2BeamLLM Encoder 对照章节。
