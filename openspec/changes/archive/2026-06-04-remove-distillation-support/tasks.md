## 1. 盘点与迁移映射

- [x] 1.1 用 `rg -n "distillation|logits_kd|\\brkd\\b|no_kd|teacher_model_name|DISTILLERS|distiller|legacy_kd" src configs tests docs README.md pyproject.toml scripts tools openspec/specs` 生成实现前命中清单。
- [x] 1.2 建立配置路径迁移表：单模态 `teacher_no_kd/student_no_kd/no_kd` 到 `strong/lightweight/supervised`，fusion canonical 到 `<slug>_strong/<slug>_lightweight`，advanced `_no_kd` 到 `_supervised` 或更具体 workflow 名称。
- [x] 1.3 建立模型 registry 迁移表：现有 strong/teacher 与 lightweight/student 注册名、公开导出名、配置字段和测试断言的替代名称。
- [x] 1.4 明确历史产物边界，确认本 change 不删除 `outputs/`、`logs/`、`All_models/`、dataset 或 archive artifact。

## 2. 配置与加载器

- [x] 2.1 将默认配置和 schema normalization 从 `model.teacher/model.student + distillation` 迁移到单主模型 `model.primary` 和 supervised/adaptation loss。
- [x] 2.2 更新 canonical/overlay recipe，生成 strong、lightweight、supervised 和 active overlay 配置，不再生成 `logits_kd`、`rkd` 或 `distillation` 字段。
- [x] 2.3 增加 migration guard，拒绝 `distillation.*` override、`logits_kd`、`rkd`、旧 `*_no_kd` 路径和旧 fusion KD virtual path，并输出新入口建议。
- [x] 2.4 删除或重命名单模态、fusion、snapshot、CSI、Raymobtime、HiST-Beam 和 MMW 源码配置中的 KD/no-KD 命名与 `distillation` block。
- [x] 2.5 更新配置 characterization 测试，覆盖新路径关键字段、旧路径拒绝、命令行覆盖顺序和缺失非 canonical YAML 的错误。

## 3. 运行时删除

- [x] 3.1 删除 `src/kd_sensing/distillation/`、distiller loss 实现、`DISTILLERS` registry 和默认组件导入中的 distillation 注册。
- [x] 3.2 移除 `engine.optim.build_distiller`、distiller optimizer 参数组、teacher checkpoint 训练解析和 trainer 中的 frozen teacher 构建/加载分支。
- [x] 3.3 简化 `BatchStepRunner` base loss：默认直接计算 supervised beam/soft-target loss，extension 继续可提供 workflow-specific base loss。
- [x] 3.4 清理训练日志、TensorBoard、run metadata、summary 和 LOSO/quick validation 中的 `distillation_enabled`、`distillation_type`、`teacher_checkpoint`、`legacy_kd` 字段写出。
- [x] 3.5 调整 `utils.artifact_registry`：保留普通 best checkpoint 和 evaluation 权重解析，删除 KD teacher checkpoint 解析入口。

## 4. 模型与工作流命名

- [x] 4.1 注册并使用 distillation-free 的 strong/lightweight 模型名称，迁移所有源码配置和测试断言。
- [x] 4.2 删除旧 teacher/student registry alias 或公开导出中的 KD 角色入口；如保留内部类名，确保公共配置和文档不再引用旧角色。
- [x] 4.3 更新 HiST-Beam、MMW、CSI、snapshot、soft beam label 和多任务 workflow，确保它们不读取 `distillation` 字段、不记录 KD loss、不声明 KD baseline。
- [x] 4.4 确保 beam soft label、V8 soft label、prototype、calibration 和 auxiliary objective 均以 supervised/adaptation 命名记录。

## 5. 文档与规格收口

- [x] 5.1 更新 README quickstart、配置矩阵、数据/产物边界和项目描述，删除 legacy KD baseline 可运行说明。
- [x] 5.2 更新 `docs/experiment_matrix.md`、`docs/extension_guide.md`、`docs/project_surface_inventory.md`、`docs/research_notes.md` 和相关脚本注释中的 KD/no-KD 入口。
- [x] 5.3 更新 `pyproject.toml` description，使项目不再以 legacy KD baseline 描述自身。
- [x] 5.4 用 `rg -n "distillation|logits_kd|\\brkd\\b|teacher_model_name|legacy_kd|DISTILLERS|distiller" src configs tests docs README.md pyproject.toml scripts tools openspec/specs` 确认只剩历史说明、拒绝错误或 archive 允许命中。

## 6. 验证

- [x] 6.1 运行 `openspec validate remove-distillation-support --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_component_registry.py tests/test_student_configs.py -q`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_prediction_objectives.py tests/test_snapshot_next_frame_baselines.py -q`。
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_lidar_modality.py tests/test_mmwave_modality.py -q`。
- [x] 6.5 运行 CLI smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`。
- [x] 6.6 最终运行 `conda run -n kd_mm_beam pytest -q`，并记录任何因外部数据或环境缺失无法完成的验证。
