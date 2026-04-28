## 1. 配置矩阵盘点

- [x] 1.1 确认 `add-lidar-modality` 的实现文件已在当前工作树中可用，并记录需要合并的 LiDAR 配置、模型和测试入口。
- [x] 1.2 枚举 canonical 单模态矩阵：`image`、`radar`、`gps`、`lidar` 各自的 `teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd`。
- [x] 1.3 枚举 canonical fusion 多模态 slug：6 个双模态、4 个三模态、1 个四模态，并确认每个 slug 的 `modalities` 固定顺序为 `image`、`radar`、`gps`、`lidar`。
- [x] 1.4 明确 legacy 配置映射表，包括 `configs/image/no_kd.yaml`、`configs/fusion/no_kd.yaml`、已有 fusion 示例配置和 radar/GPS/LiDAR 的 `no_kd.yaml`。

## 2. 单模态配置

- [x] 2.1 新增 `configs/image/teacher_no_kd.yaml`，使用 `image_teacher` 作为 no-KD 可训练主模型，并设置 canonical run name。
- [x] 2.2 新增或调整 `configs/image/student_no_kd.yaml`，使用 `image_student` 作为 no-KD 可训练主模型，并让 legacy `configs/image/no_kd.yaml` 保持兼容语义。
- [x] 2.3 为 `configs/radar/`、`configs/gps/`、`configs/lidar/` 新增 `teacher_no_kd.yaml` canonical 入口，并保持现有 `no_kd.yaml` 作为 teacher baseline 兼容入口。
- [x] 2.4 检查并统一所有单模态 `student_no_kd.yaml`，确保 `model.student.type` 为 lightweight student、`distillation.teacher_model_name: null`、`experiment.name` 与 `output.run_name` 一致。
- [x] 2.5 更新所有单模态 `logits_kd.yaml` 和 `rkd.yaml` 的默认 teacher checkpoint 来源，使 canonical 配置默认指向对应 `teacher_no_kd` 输出目录，同时保留命令行覆盖能力。
- [x] 2.6 确认所有单模态 teacher/student `gru_params` 为 `[64, 64, 2]`，GPS 配置使用 `relative_polar` 与 `gps_input_size: 3`，LiDAR 配置使用 BEV 默认字段。

## 3. Fusion 配置

- [x] 3.1 为 11 个多模态 slug 新增 `<slug>_teacher_no_kd.yaml`，使用 `fusion_teacher` 作为 no-KD 可训练主模型。
- [x] 3.2 为 11 个多模态 slug 新增 `<slug>_student_no_kd.yaml`，使用 `fusion_student` 作为 no-KD 可训练主模型。
- [x] 3.3 为 11 个多模态 slug 新增 `<slug>_logits_kd.yaml`，使用 frozen `fusion_teacher` 蒸馏可训练 `fusion_student`。
- [x] 3.4 为 11 个多模态 slug 新增 `<slug>_rkd.yaml`，使用 frozen `fusion_teacher` 蒸馏可训练 `fusion_student`，并配置 RKD 参数。
- [x] 3.5 为所有包含 GPS 的 fusion 配置设置 `use_gps: true`、`gps_feature_mode: relative_polar` 和 teacher/student `gps_input_size: 3`。
- [x] 3.6 为所有包含 LiDAR 的 fusion 配置设置 `use_lidar: true`、LiDAR BEV 默认字段和 teacher/student `lidar_channels`。
- [x] 3.7 保留现有 `configs/fusion/no_kd.yaml`、`logits_kd.yaml`、`rkd.yaml` 和已有示例配置作为兼容入口，并确保它们仍能按原有显式 `modalities` 构建。

## 4. 文档

- [x] 4.1 更新 README 训练命令，优先展示 canonical 单模态与 fusion 配置矩阵，并说明 teacher baseline -> KD 的推荐运行顺序。
- [x] 4.2 在 README 中说明 legacy `no_kd.yaml` 的历史语义差异，以及推荐使用 `teacher_no_kd.yaml` / `student_no_kd.yaml` 的原因。
- [x] 4.3 更新 README 中原仓库九个权重的归类说明，明确 teacher-as-student 残留不作为当前配置驱动流程的默认语义。
- [x] 4.4 如 `docs/extension_guide.md` 引用了配置命名或 fusion `modalities`，同步更新为 canonical 命名和完整多模态组合规则。

## 5. 测试

- [x] 5.1 更新 `tests/test_student_configs.py`，枚举并验证所有 canonical 单模态配置存在且模型角色、distillation 类型、run name 和 checkpoint 来源正确。
- [x] 5.2 新增或更新 fusion 配置矩阵测试，验证 11 个多模态 slug 的四类 canonical 配置存在，teacher/student `modalities` 一致且数据字段启用正确。
- [x] 5.3 保留 legacy 配置测试，确保旧 `no_kd.yaml` 和现有 fusion 示例入口仍能构建，并明确它们映射到哪个 canonical 语义。
- [x] 5.4 更新 GPS 和 LiDAR 相关测试，覆盖包含 GPS/LiDAR 的 fusion canonical 配置构建与缺失输入错误路径。
- [x] 5.5 更新 legacy 预训练权重兼容测试，确保 `All_models/*Std*.pth` 的一层 GRU 兼容说明仍被测试覆盖。

## 6. 验证

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py`，验证配置矩阵和 legacy 权重兼容测试。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_gps_modality.py tests/test_lidar_modality.py`，验证 GPS/LiDAR 与 fusion 输入路径。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest`，确认完整测试套件通过。
- [x] 6.4 运行 `openspec status --change standardize-experiment-config-matrix`，确认变更 artifacts 和任务状态可进入实施阶段。
