## 1. 模型实现

- [x] 1.1 在 `src/kd_sensing/models/radar.py` 中新增 `RadarStudentNet`，注册名为 `radar_student`，构造参数支持 `feature_size`、`num_classes`、`gru_params`、`radar_channels`
- [x] 1.2 为 `RadarStudentNet` 实现 depthwise separable radar CNN、adaptive avg/max pooling、feature projection、LayerNorm、GRU 和 classifier
- [x] 1.3 在 `RadarStudentNet` 中校验 `gru_params` 长度和 `gru_input_size == feature_size`
- [x] 1.4 确保 `RadarStudentNet.forward()` 接收 `(batch, sequence, channels, height, width)` 雷达张量，并返回 `(pred, features, output_features)`
- [x] 1.5 更新 `src/kd_sensing/models/__init__.py` 导出 `RadarStudentNet`

## 2. 配置更新

- [x] 2.1 将 `configs/radar/logits_kd.yaml` 的 `model.student.type` 改为 `radar_student`，并使用默认 `[64, 64, 1]` student GRU 参数
- [x] 2.2 将 `configs/radar/rkd.yaml` 的 `model.student.type` 改为 `radar_student`，并保持 teacher checkpoint 默认来源为 `outputs/radar_no_kd/checkpoints/best.pth`
- [x] 2.3 新增 `configs/radar/student_no_kd.yaml`，使用 `radar_student` 作为 no-KD 可训练主模型，且 `distillation.teacher_model_name: null`
- [x] 2.4 保留 `configs/radar/no_kd.yaml` 的 `radar_teacher` baseline 语义，用于训练 RadarTeacher checkpoint

## 3. 测试更新

- [x] 3.1 更新 `tests/test_student_configs.py`，断言 radar KD 配置构建 `RadarStudentNet` student 和 `RadarTeacherNet` teacher
- [x] 3.2 新增 RadarStudent forward contract 测试，验证 logits、input features 和 output features 的 batch/sequence/feature 形状
- [x] 3.3 新增 RadarStudent 参数校验测试，覆盖非法 `gru_params` 和 input size mismatch
- [x] 3.4 更新 radar no-KD 配置测试，区分 `radar_teacher` baseline 和 `radar_student` no-KD 配置

## 4. 文档更新

- [x] 4.1 更新 README radar-only 训练说明，区分 `configs/radar/no_kd.yaml` 的 teacher baseline 和 `configs/radar/student_no_kd.yaml` 的 lightweight student no-KD
- [x] 4.2 更新 README radar KD 说明，明确 `logits_kd` 和 `rkd` 默认使用 frozen `radar_teacher` 蒸馏可训练 `radar_student`

## 5. 验证

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py` 验证模型构建、配置和 forward contract
- [x] 5.2 运行 `openspec status --change add-radar-student` 确认 change artifacts 处于可实施状态
