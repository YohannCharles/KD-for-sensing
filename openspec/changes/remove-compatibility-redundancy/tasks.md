## 1. 规格与引用基线

- [ ] 1.1 更新 active OpenSpec 变更中仍要求 legacy/compat 行为的 proposal、design、tasks 和 specs，确保它们不再要求保留 `scenario9` dataset type、compat facade、legacy fusion 配置或 legacy weight fallback。
- [ ] 1.2 使用 `rg -n "Scenario9Dataset|Scenario31Dataset|Scenario32Dataset|data\\.dataset\\.type: scenario9|kd_sensing\\.engine\\.builders|kd_sensing\\.data\\.transforms|transform_ops\\._legacy|configs/fusion/(no_kd|logits_kd|rkd)\\.yaml|FusionModalityNet|StudentModalityNet|legacy_path|legacy weight|kd-sensing-visualize-modalities" src configs tests docs README.md openspec/changes --glob '!openspec/changes/archive/**'` 建立当前可执行引用清单。
- [ ] 1.3 标记每个引用的处理方式：迁移到 canonical 路径、删除旧测试、改为旧入口拒绝测试，或保留为历史文档但不得作为运行入口。

## 2. DeepSense6G Dataset 收敛

- [ ] 2.1 将 `src/kd_sensing/data/datasets/scenario9.py` 中仍在使用的 DeepSense6G dataset 主实现迁移到场景中立模块，例如 `src/kd_sensing/data/datasets/deepsense6g.py`。
- [ ] 2.2 更新 dataset 注册和默认组件导入，只注册 `deepsense6g`，删除 `scenario9`、`scenario31`、`scenario32` dataset 注册名。
- [ ] 2.3 更新 `src/kd_sensing/data/datasets/__init__.py`、`src/kd_sensing/data/__init__.py` 和 `src/kd_sensing/registries.py`，不再导出或导入 `Scenario9Dataset`、`Scenario31Dataset`、`Scenario32Dataset` 或 `kd_sensing.data.datasets.scenario9`。
- [ ] 2.4 更新 `src/kd_sensing/data/scenes.py` 和配置标准化逻辑，拒绝 `data.dataset.type: scenario9|scenario31|scenario32`，错误信息指向 `data.dataset.type: deepsense6g` 加 `data.dataset.scene`。
- [ ] 2.5 重写 dataset 相关测试，直接构建 `DeepSense6GDataset` 或通过 `DATASETS.build("deepsense6g")` 选择场景，删除对 `Scenario9Dataset` 的直接依赖。

## 3. 兼容 Facade 与内部导入清理

- [ ] 3.1 删除 `src/kd_sensing/engine/builders.py` 和任何 `_builders_impl` 兼容聚合残留，把内部和测试导入迁移到 `engine.data_factory`、`engine.optim`、`engine.cache_policy`、`engine.normalization_artifacts`、`engine.run_metadata` 等窄模块。
- [ ] 3.2 删除 `src/kd_sensing/data/transforms.py` 和 `src/kd_sensing/data/transform_ops/_legacy.py`，把内部和测试导入迁移到 `data.transform_ops.<modality>` 或通用 transform 模块。
- [ ] 3.3 更新 `__init__.py` 延迟导出逻辑，确保包级轻量导入不通过已删除 compat facade 间接加载重依赖。
- [ ] 3.4 增强架构边界测试，验证内部代码不再引用 `kd_sensing.engine.builders`、`kd_sensing.engine._builders_impl`、`kd_sensing.data.transforms` 或 `kd_sensing.data.transform_ops._legacy`。

## 4. 配置、模型 Alias 与 Artifact Registry

- [ ] 4.1 删除或重命名 legacy fusion 配置路径 `configs/fusion/no_kd.yaml`、`configs/fusion/logits_kd.yaml` 和 `configs/fusion/rkd.yaml`，保留或生成对应 canonical `image_radar_*` 路线。
- [ ] 4.2 更新配置 loader 和测试，禁止把旧 legacy fusion 路径静默映射到 canonical 配置；旧路径缺失或被拒绝时必须给出迁移提示。
- [ ] 4.3 删除旧 fusion 类名 alias，例如 `FusionModalityNet`、`StudentModalityNet`，保留 canonical `FusionTeacherModalityNet` 和 `FusionStudentModalityNet`。
- [ ] 4.4 删除 legacy image/model/encoder alias 测试期望，保留当前 canonical 注册名的构建和错误诊断测试。
- [ ] 4.5 更新 artifact registry 和 checkpoint 解析，删除 legacy weight fallback；registry 缺失时只报告 registry 候选和显式 checkpoint 配置方式。
- [ ] 4.6 更新训练、评估和 KD 配置测试，确认没有 `legacy_path` 或 legacy weight fallback 被记录到新的运行 metadata。

## 5. 诊断入口与文档

- [ ] 5.1 删除旧可视化兼容入口或将其从安装脚本和推荐命令中移除，只保留 `kd-sensing-export-viewer-manifest` 与 Gradio viewer 路线。
- [ ] 5.2 更新 README、`docs/extension_guide.md`、诊断说明和训练说明，删除“旧入口继续兼容”的表述，补充 canonical 迁移路径。
- [ ] 5.3 确认文档中只把 `data.dataset.scene` 的 `scenario9` 等字符串作为场景别名描述，不再把 `scenario9` 描述为 dataset type。
- [ ] 5.4 若保留 `All_models/` 历史文件，文档必须说明它们不是默认 runtime fallback；若决定删除文件，需要单独列出文件清理和大文件影响。

## 6. 测试与验证

- [ ] 6.1 使用 `conda run -n kd_mm_beam pytest -q tests/test_architecture_boundaries.py` 验证导入边界和 compat facade 删除。
- [ ] 6.2 使用 `conda run -n kd_mm_beam pytest -q tests/test_training_io_workflow.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py` 验证 DeepSense6G dataset、场景选择和模态按需读取。
- [ ] 6.3 使用 `conda run -n kd_mm_beam pytest -q tests/test_student_configs.py tests/test_craf_fusion.py tests/test_teacher_prior_craf.py` 验证 fusion canonical 配置、旧 alias 删除和 CRAF/G2D 相关配置不回退到 legacy 路线。
- [ ] 6.4 使用 `conda run -n kd_mm_beam pytest -q tests/test_modality_visual_diagnostics.py` 验证 viewer manifest 路线和旧可视化入口清理。
- [ ] 6.5 使用 `rg -n "Scenario9Dataset|Scenario31Dataset|Scenario32Dataset|data\\.dataset\\.type: scenario9|kd_sensing\\.engine\\.builders|kd_sensing\\.data\\.transforms|transform_ops\\._legacy|configs/fusion/(no_kd|logits_kd|rkd)\\.yaml|FusionModalityNet|StudentModalityNet|legacy_path|legacy weight|kd-sensing-visualize-modalities" src configs tests docs README.md openspec/changes --glob '!openspec/changes/archive/**'` 确认可执行代码、测试、文档和 active OpenSpec 不再残留旧兼容引用。
- [ ] 6.6 使用 `conda run -n kd_mm_beam pytest -q` 运行全量回归。
- [ ] 6.7 使用 `openspec status --change remove-compatibility-redundancy` 确认变更 apply-ready，并记录仍需人工决策的问题。
