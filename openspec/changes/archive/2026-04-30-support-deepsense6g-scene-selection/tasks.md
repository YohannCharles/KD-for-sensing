## 1. 场景解析与配置

- [x] 1.1 新增 DeepSense6G 场景描述符和解析工具，支持 `9`、`scene9`、`scenario9`、`32`、`scene32`、`scenario32`，未知场景给出支持列表。
- [x] 1.2 扩展默认配置，增加 `data.dataset.scene: 32` 和 `output.group_by_scene: true`，并让缺失的 `data_root`、train/test CSV 能由场景默认值补齐。
- [x] 1.3 在配置校验或 dataset 构建前统一规范化场景字段，确保最终配置包含 `scene_id`、`scene_slug` 和解析后的数据路径。

## 2. Dataset 构建与 metadata

- [x] 2.1 将 Scenario 9 数据集读取实现抽象为通用 DeepSense6G dataset，同时保留 `Scenario9Dataset` 导出和 `scenario9` registry 兼容。
- [x] 2.2 注册通用 `deepsense6g` dataset 类型，并添加 `scenario32` 兼容注册名。
- [x] 2.3 更新 `build_dataset()` 和 split metadata，使 train/test dataset 都记录 `scene_id`、`scene_slug`、CSV 路径、样本数和启用模态。
- [x] 2.4 补充 dataset 测试，覆盖旧 `scenario9` 配置、`deepsense6g + scene=32`、场景别名和未知场景错误。

## 3. 输出目录与 registry

- [x] 3.1 更新训练运行目录解析，使默认 DeepSense6G 训练输出写入 `outputs/<scene_slug>/<run_name>/`，并保持 `resume`、`overwrite` 和唯一目录后缀语义。
- [x] 3.2 更新默认评估目录解析，使配置驱动评估按场景分组，但显式 `--output-dir` 作为完整路径使用。
- [x] 3.3 更新 checkpoint registry 默认目录解析为当前场景下的 `best_checkpoints`，并在 sidecar metadata 中记录场景字段。
- [x] 3.4 更新 KD teacher fallback 解析和 canonical KD YAML，使默认权重路径指向当前场景目录；新默认配置指向 `outputs/scene32/<teacher_run_name>/checkpoints`，历史 Scenario 9 配置指向 `outputs/scene9/<teacher_run_name>/checkpoints`。
- [x] 3.5 补充训练目录、评估目录、registry 隔离和 KD teacher fallback 单元测试。

## 4. 配置、文档与迁移

- [x] 4.1 批量更新 `configs/**.yaml` 中的默认 DeepSense6G 配置，加入 `data.dataset.scene: 32`，移除或同步旧硬编码 `dataset/scenario9`，并保留 `scenario9` 作为显式兼容入口。
- [x] 4.2 更新 README 和相关文档，说明默认场景为 Scenario 32、`data.dataset.scene=9` 的覆盖方式、默认输出目录和历史输出迁移规则。
- [x] 4.3 将当前 `outputs/<run_name>/` 训练目录迁移到 `outputs/scene9/<run_name>/`，将 `outputs/best_checkpoints/` 迁移到 `outputs/scene9/best_checkpoints/`，避免静默覆盖。
- [x] 4.4 检查迁移后的 registry sidecar JSON，必要时更新其中的 `path`、`run_dir`、`scene_id` 和 `scene_slug` 字段。

## 5. 验证

- [x] 5.1 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_student_configs.py tests/test_gps_modality.py tests/test_mmwave_modality.py` 验证配置、目录、registry 和数据集行为。
- [x] 5.2 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/mmwave/teacher_no_kd.yaml --dry-run` 验证 scene32 默认训练输出路径。
- [x] 5.3 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/mmwave/teacher_no_kd.yaml --dry-run data.dataset.scene=9 output.run_name=scene9_dry_run` 验证命令行场景覆盖和 scene9 输出路径。
- [x] 5.4 运行 `openspec status --change support-deepsense6g-scene-selection` 确认变更 apply-ready。
