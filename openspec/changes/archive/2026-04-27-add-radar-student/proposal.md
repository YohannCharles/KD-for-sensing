## Why

当前 radar-only 工作流只有 `radar_teacher`，KD 配置中 teacher 和 student 使用同一套重模型，和 image-only、fusion 工作流中“冻结 teacher + 轻量 student”的设计不一致。新增轻量 `radar_student` 可以让 radar-only 蒸馏真正训练小模型，同时保留 `radar_teacher` 作为 teacher checkpoint 和 radar baseline。

## What Changes

- 新增已注册模型 `radar_student`，输入仍为 RA/DA 拼接后的雷达序列，输出保持 `(pred, features, output_features)` 契约。
- `radar_student` 使用轻量 CNN embedding、全局池化、特征投影、LayerNorm、GRU 和小型 classifier，结构风格参考 `image_student` 与 `fusion_student` 的 radar 分支。
- radar KD 配置默认使用 frozen `radar_teacher` 作为 teacher，使用可训练 `radar_student` 作为 student。
- 保留 `radar_teacher` 和现有 radar teacher baseline 语义，用于训练或加载 teacher checkpoint。
- 增加 radar student 构建、前向输出、配置解析和权重兼容相关回归测试。
- 更新 README 或扩展说明，明确 radar teacher baseline、radar student no-KD 和 radar KD 的推荐使用方式。

## Capabilities

### New Capabilities
- `radar-student-model`: 定义 radar-only 轻量 student 的模型结构、注册名称、前向输出契约和配置构建要求。

### Modified Capabilities
- `component-registry`: 增加 radar-only student 通过 `MODELS` 注册表构建并复用 radar-only 流程的要求。
- `experiment-workflow`: 调整 radar-only KD 默认配置，使其构建 `radar_teacher` teacher 和 `radar_student` student，并补充轻量 radar student 的 no-KD 训练入口。
- `radar-teacher-model`: 调整 RadarTeacher 蒸馏角色说明，明确默认 KD student 不再复用 teacher 架构，但仍保留显式配置 teacher-as-student 的兼容能力。

## Impact

- 影响模型代码：`src/kd_sensing/models/radar.py` 新增 `RadarStudentNet` 并注册为 `radar_student`，`src/kd_sensing/models/__init__.py` 更新导出。
- 影响配置：`configs/radar/logits_kd.yaml`、`configs/radar/rkd.yaml` 的 student 指向 `radar_student`；新增或更新 radar student no-KD 配置；保留 teacher checkpoint 默认来源。
- 影响测试：更新 `tests/test_student_configs.py`，新增 radar student forward contract、配置构建和 KD 配置断言。
- 影响文档：README 中 radar-only 配置说明需要区分 teacher baseline 与 lightweight student 实验。
