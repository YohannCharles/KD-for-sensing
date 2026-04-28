## 1. 配置统一

- [x] 1.1 将 `src/kd_sensing/config/defaults.py` 中默认 teacher 和 student 的 `gru_params` 改为 `[64, 64, 2]`。
- [x] 1.2 将 `configs/image/*.yaml` 中所有 `model.teacher.gru_params` 和 `model.student.gru_params` 改为 `[64, 64, 2]`。
- [x] 1.3 将 `configs/radar/*.yaml` 中所有 `model.teacher.gru_params` 和 `model.student.gru_params` 改为 `[64, 64, 2]`。
- [x] 1.4 将 `configs/gps/*.yaml` 中所有 `model.teacher.gru_params` 和 `model.student.gru_params` 改为 `[64, 64, 2]`。
- [x] 1.5 将 `configs/fusion/*.yaml` 中所有 `model.teacher.gru_params` 和 `model.student.gru_params` 改为 `[64, 64, 2]`。

## 2. Radar 模型命名

- [x] 2.1 在 `src/kd_sensing/models/radar.py` 中将旧 radar teacher 类重命名为 `RadarModalityNet`，并将 `self.name` 更新为 `"RadarModalityNet"`。
- [x] 2.2 在 `src/kd_sensing/models/radar.py` 中将旧 radar student 类重命名为 `RadarStudentModalityNet`，并将 `self.name` 更新为 `"RadarStudentModalityNet"`。
- [x] 2.3 保留 `@MODELS.register("radar_teacher")` 和 `@MODELS.register("radar_student")` 注册名不变，确保现有 YAML 配置继续可构建。
- [x] 2.4 更新 `src/kd_sensing/models/__init__.py` 的导入和 `__all__`，导出 `RadarModalityNet` 与 `RadarStudentModalityNet`。

## 3. 测试更新

- [x] 3.1 更新 `tests/test_student_configs.py` 中 radar 类导入、`isinstance` 断言和所有默认 `gru_params` 断言，使 image/radar/GPS/fusion 默认 teacher/student 均为 `[64, 64, 2]`。
- [x] 3.2 更新 `tests/test_student_configs.py` 中 radar teacher/student forward 和参数校验测试，使用 `RadarModalityNet` 与 `RadarStudentModalityNet`。
- [x] 3.3 更新 `tests/test_gps_modality.py` 中示例配置和断言，使默认 GPS student 使用二层 GRU。
- [x] 3.4 检查测试中是否仍引用旧 radar 类名或默认一层 GRU 的受支持配置断言，并完成迁移。

## 4. 文档更新

- [x] 4.1 更新 `README.md` 中关于 teacher/student 架构、checkpoint 兼容性和 radar 类名的说明，明确配置注册名仍为 `radar_teacher`/`radar_student`。
- [x] 4.2 更新 `docs/extension_guide.md` 中的 `gru_params` 示例和 radar 类名说明，使默认示例使用 `[64, 64, 2]`。
- [x] 4.3 使用 `rg` 检查仓库内旧 radar 类名和默认一层 GRU 参数引用，并只保留明确标注为历史归档的内容。

## 5. 验证

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_gps_modality.py`，确认配置构建、radar/GPS forward 和参数校验测试通过。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest`，确认完整测试套件通过。
- [x] 5.3 运行 `openspec status --change "standardize-gru-and-radar-naming"`，确认变更保持 apply-ready。
