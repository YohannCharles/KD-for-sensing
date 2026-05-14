## 1. 规格与引用基线

- [x] 1.1 更新 active OpenSpec proposal、design、tasks 和 specs，使它们只描述删除/拒绝语义，不再要求保留旧 dataset type、compat facade、旧 fusion 配置别名或 checkpoint 目录 fallback。
- [x] 1.2 建立旧入口引用基线，并确认命中项分布在 README/docs、测试、dataset 注册、builder/transform facade、fusion alias、artifact registry 和本 change 的删除说明中。
- [x] 1.3 标记每类引用的处理方式：迁移到 canonical 路径、删除旧测试、改为旧入口拒绝测试，或保留为历史说明但不得作为运行入口。

## 2. DeepSense6G Dataset 收敛

- [x] 2.1 将旧场景命名文件中的 DeepSense6G dataset 主实现迁移到 `src/kd_sensing/data/datasets/deepsense6g.py`。
- [x] 2.2 更新 dataset 注册和默认组件导入，只注册 `deepsense6g`，删除场景专用 dataset 注册名。
- [x] 2.3 更新 `src/kd_sensing/data/datasets/__init__.py`、`src/kd_sensing/data/__init__.py` 和 `src/kd_sensing/registries.py`，不再导出或导入场景专用 dataset class alias。
- [x] 2.4 更新 `src/kd_sensing/data/scenes.py` 和配置标准化逻辑，拒绝场景专用 dataset type，错误信息指向 `deepsense6g` 加 `data.dataset.scene`。
- [x] 2.5 重写 dataset 相关测试，直接构建 `DeepSense6GDataset` 或通过 `DATASETS.build("deepsense6g")` 选择场景。

## 3. 兼容 Facade 与内部导入清理

- [x] 3.1 删除 builder 聚合 facade 和私有聚合残留，把内部和测试导入迁移到 `engine.data_factory`、`engine.optim`、`engine.cache_policy`、`engine.normalization_artifacts`、`engine.run_metadata` 等窄模块。
- [x] 3.2 删除 transform 聚合 facade，把内部和测试导入迁移到 `data.transform_ops.<modality>` 或通用 transform 子模块。
- [x] 3.3 更新 `__init__.py` 延迟导出逻辑，确保包级轻量导入不通过已删除 compat facade 间接加载重依赖。
- [x] 3.4 增强架构边界测试，验证内部代码不再引用已删除的 builder/transform 聚合入口。

## 4. 配置、模型 Alias 与 Artifact Registry

- [x] 4.1 删除旧 fusion 三个顶层配置别名，保留对应 canonical `image_radar_*` 路线。
- [x] 4.2 更新配置 loader 和测试，禁止把旧 fusion 路径静默映射到 canonical 配置；旧路径缺失或被拒绝时给出迁移提示。
- [x] 4.3 删除旧 fusion 类名 alias，保留职责明确的 canonical teacher/student fusion 类和注册名。
- [x] 4.4 删除旧 image/model/encoder alias 测试期望，保留当前 canonical 注册名的构建和错误诊断测试。
- [x] 4.5 更新 artifact registry 和 checkpoint 解析，删除 checkpoint 目录 fallback；registry 缺失时只报告 registry 候选和显式 checkpoint 配置方式。
- [x] 4.6 更新训练、评估和 KD 配置测试，确认新的运行 metadata 不再记录已删除的 fallback 字段。

## 5. 诊断入口与文档

- [x] 5.1 删除旧可视化 console/script 入口，只保留 `kd-sensing-export-viewer-manifest` 与 Gradio viewer 路线。
- [x] 5.2 更新 README、`docs/extension_guide.md`、诊断说明和训练说明，删除“旧入口继续兼容”的表述，补充 canonical 迁移路径。
- [x] 5.3 确认文档中只把 `data.dataset.scene` 的 `scenario9` 等字符串作为场景别名描述，不再把 `scenario9` 描述为 dataset type。
- [x] 5.4 保留 `All_models/` 历史文件，并在文档中说明它们不是默认 runtime fallback。

## 6. 测试与验证

- [x] 6.1 使用 `conda run -n kd_mm_beam pytest -q tests/test_architecture_boundaries.py` 验证导入边界和 compat facade 删除。
- [x] 6.2 使用 `conda run -n kd_mm_beam pytest -q tests/test_training_io_workflow.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py` 验证 DeepSense6G dataset、场景选择和模态按需读取。
- [x] 6.3 使用 `conda run -n kd_mm_beam pytest -q tests/test_student_configs.py tests/test_craf_fusion.py tests/test_teacher_prior_craf.py` 验证 fusion canonical 配置、旧 alias 删除和 CRAF/G2D 相关配置不回退到旧路线。
- [x] 6.4 使用 `conda run -n kd_mm_beam pytest -q tests/test_modality_visual_diagnostics.py` 验证 viewer manifest 路线和旧可视化入口清理。
- [x] 6.5 运行旧入口引用扫描，确认可执行代码、测试、文档和 active OpenSpec 不再残留已删除的运行入口。
- [x] 6.6 使用 `conda run -n kd_mm_beam pytest -q` 运行全量回归。
- [x] 6.7 使用 `openspec status --change remove-compatibility-redundancy` 确认变更 apply-ready，并记录仍需人工决策的问题。
