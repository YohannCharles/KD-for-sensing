## 1. Fusion 模型命名

- [x] 1.1 将 legacy fusion teacher 实现公开为 `FusionTeacherModalityNet`，并让 `fusion_teacher` 注册名构建该类。
- [x] 1.2 将 legacy fusion student 实现公开为 `FusionStudentModalityNet`，并让 `fusion_student` 注册名构建该类。
- [x] 1.3 保留 `FusionModalityNet` 和 `StudentModalityNet` 作为兼容 alias，确保旧 import 不破坏。
- [x] 1.4 更新 `kd_sensing.models`、`kd_sensing.models.fusion` 导出和相关类型断言，新代码优先使用新类名。

## 2. 场景中立配置命名

- [x] 2.1 将 `configs/fusion/scene32_*.yaml` 重命名为不含 `scene32_` 前缀的配置文件。
- [x] 2.2 同步更新这些配置中的 `experiment.name` 和 `output.run_name`，确保不再出现 `scene32_` 前缀。
- [x] 2.3 更新引用旧配置路径的 README、tools 文档、analysis 配置和测试。
- [x] 2.4 确认 dataset 场景字段、输出场景目录和 checkpoint metadata 语义保持不变。

## 3. 兼容层与入口收敛

- [x] 3.1 将内部脚本中仍使用 `kd_sensing.engine.builders` 的地方切到 `engine.data_factory`、`engine.optim` 或 `engine.run_metadata` 窄模块。
- [x] 3.2 将 viewer manifest orchestration 从重复 tools 脚本切到 `kd-sensing-export-viewer-manifest` 或包内 CLI。
- [x] 3.3 评估 `engine._builders_impl` 和 `data.transform_ops._legacy` 是否还有内部依赖；如无必要依赖，删除或保留为仅外部兼容的最小 re-export。
- [x] 3.4 更新架构边界测试，防止新增内部代码依赖二级兼容聚合层。

## 4. 验证

- [x] 4.1 使用 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_fusion_image_feature_extractor.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py -q` 验证 fusion 命名和配置矩阵。
- [x] 4.2 使用 `conda run -n kd_mm_beam pytest tests/test_craf_fusion.py tests/test_marf_fusion.py tests/test_marf_training.py tests/test_teacher_prior_craf.py -q` 验证 CRAF/MARF 配置改名。
- [x] 4.3 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证导入边界。
- [x] 4.4 使用 `openspec validate --all` 和 `openspec status --change standardize-fusion-naming-and-scene-neutral-configs` 验证 OpenSpec 状态。
