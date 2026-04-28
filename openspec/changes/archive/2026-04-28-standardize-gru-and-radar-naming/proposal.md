## Why

当前各模态默认 `gru_params` 不一致，image-only 和部分 student 配置仍使用一层 GRU，且与论文中二层 GRU 的描述以及 fusion/radar teacher 配置不统一。`radar.py` 的公开类名也使用旧的 teacher/student Net 风格命名，和 image、GPS 中的 `*ModalityNet` 命名风格不一致，增加了模型代码、测试和文档的认知成本。

## What Changes

- 将所有受支持的单模态和多模态实验配置中的 `gru_params` 统一为 `[64, 64, 2]`，覆盖 teacher 与 student、no-KD 与 KD 配置、默认配置和文档示例。
- 更新配置测试和模型构建测试，断言 image、radar、GPS 和 fusion 的默认 teacher/student GRU 层数均为 2。
- 将 radar 模型实现的公开类名改为与其它模态一致的命名：
  - teacher 类改为 `RadarModalityNet`
  - student 类改为 `RadarStudentModalityNet`
- 保留 `MODELS` 注册名 `radar_teacher` 和 `radar_student`，使现有 YAML 配置入口继续可用。
- 更新 `src/kd_sensing/models/__init__.py`、测试、README、扩展指南和 OpenSpec 规格中对 radar 类名与默认 GRU 层数的描述。
- **BREAKING**：现有一层 GRU 的 image/GPS/radar/fusion student checkpoint 与新的二层 GRU 配置不完全兼容；按新配置复现实验时需要重新训练或使用对应二层 GRU 权重。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `experiment-workflow`: 默认实验配置必须统一使用 `[64, 64, 2]` 作为所有单模态和多模态 teacher/student 的 `gru_params`。
- `radar-teacher-model`: `radar_teacher` 的公开实现类名和导出应为 `RadarModalityNet`，同时保留 `radar_teacher` 注册名与输入输出契约。
- `radar-student-model`: `radar_student` 的公开实现类名和导出应为 `RadarStudentModalityNet`，默认 `gru_params` 改为 `[64, 64, 2]`，同时保留轻量 student 语义。
- `gps-modality-model`: GPS student 描述不再要求单层 GRU，默认 GPS teacher/student 配置必须使用 `[64, 64, 2]` 并保持 KD 输出维度兼容。

## Impact

- 影响配置：`configs/image/*.yaml`、`configs/radar/*.yaml`、`configs/gps/*.yaml`、`configs/fusion/*.yaml`、`src/kd_sensing/config/defaults.py`。
- 影响模型代码：`src/kd_sensing/models/radar.py` 和 `src/kd_sensing/models/__init__.py`。
- 影响测试：配置构建测试、radar forward/参数校验测试、GPS/fusion 相关 GRU 层数断言。
- 影响文档：README、扩展指南以及 OpenSpec radar/GPS/实验流程规格。
- 不新增第三方依赖，不改变训练/验证/评估入口，不改变配置注册名 `radar_teacher` 和 `radar_student`。
