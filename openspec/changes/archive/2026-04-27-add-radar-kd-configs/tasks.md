## 1. 配置补齐

- [x] 1.1 新增 `configs/radar/logits_kd.yaml`，基于 `configs/radar/no_kd.yaml` 保持 radar dataset、RadarTeacher teacher/student、训练超参数和输出结构一致，并设置 `distillation.type: logits_kd`、`temperature: 3.0`、`alpha: 0.4`、`alpha_warmup_epochs: 0`。
- [x] 1.2 新增 `configs/radar/rkd.yaml`，保持 radar 基础配置一致，并设置 `distillation.type: rkd`、logits KD 通用参数和 `rkd_pairs_per_anchor: 4`、`rkd_distance_weight: 10.0`、`rkd_angle_weight: 10.0`。
- [x] 1.3 在两个 radar KD 配置中设置默认 teacher 权重来源为 `paths.weights_dir: outputs/radar_no_kd/checkpoints` 与 `distillation.teacher_model_name: best.pth`，确保默认解析到 `outputs/radar_no_kd/checkpoints/best.pth`。

## 2. 文档和回归测试

- [x] 2.1 更新 `README.md` 的训练命令，列出 `configs/radar/no_kd.yaml`、`configs/radar/logits_kd.yaml` 和 `configs/radar/rkd.yaml`。
- [x] 2.2 更新 README 的 radar-only 说明，写明 `no_kd`、`logits_kd`、`rkd` 的区别，以及 radar KD 默认依赖先训练出的 `outputs/radar_no_kd/checkpoints/best.pth`。
- [x] 2.3 更新 `tests/test_student_configs.py` 的 radar 配置列表，覆盖三套 radar 配置并验证它们都构建 `RadarTeacherNet`。
- [x] 2.4 增加或调整 radar 配置断言：`no_kd` 不加载 teacher；`logits_kd` 和 `rkd` 设置正确的 distillation type、teacher checkpoint 字段；`rkd` 包含 RKD pair/距离/角度权重。

## 3. 验证

- [x] 3.1 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py` 运行配置回归测试。
- [x] 3.2 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/radar/no_kd.yaml --dry-run` 生成临时 radar no-KD checkpoint，并确认 `outputs/radar_no_kd_dry_run/checkpoints/best.pth` 存在。
- [x] 3.3 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/radar/logits_kd.yaml --dry-run --override paths.weights_dir=outputs/radar_no_kd_dry_run/checkpoints --override distillation.teacher_model_name=best.pth` 验证 radar logits KD 训练路径可完成。
- [x] 3.4 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/radar/rkd.yaml --dry-run --override paths.weights_dir=outputs/radar_no_kd_dry_run/checkpoints --override distillation.teacher_model_name=best.pth` 验证 radar RKD 训练路径可完成。
