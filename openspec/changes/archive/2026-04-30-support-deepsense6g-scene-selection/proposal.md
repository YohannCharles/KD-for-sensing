## Why

当前 DeepSense6G 数据路径和实验输出都隐含固定在 Scenario 9：配置大量写死 `scenario9` 与 `dataset/scenario9`，训练产物也直接落在 `outputs/<run_name>/`。现在需要同时支持 Scenario 9 和 Scenario 32，并为未来新增场景保留扩展点；否则不同场景的训练结果、checkpoint 和 teacher 权重容易混在一起，影响复现实验和横向比较。

## What Changes

- 将 DeepSense6G 场景从固定 `scenario9` 扩展为可配置字段，首批支持 `scene9`/`9` 和 `scene32`/`32`，后续可继续注册更多场景。
- 将新默认场景改为 Scenario 32；旧的 `type: scenario9` 配置仍兼容并解析为 Scenario 9。
- 让 dataset 构建流程能根据场景解析默认 `data_root`、train/test CSV 和 dataset type，仍允许用户显式覆盖数据根目录和 CSV 文件名。
- 将训练与评估输出默认按场景归类，例如 `outputs/scene9/<run_name>/`、`outputs/scene32/<run_name>/`。
- 将现有历史训练输出统一迁移到 `scene9` 目录；新默认训练、checkpoint registry 和 KD teacher 默认权重解析按 Scenario 32 归入 `scene32`，显式绝对路径或显式输出目录覆盖保持最高优先级。
- 更新 README、配置示例和测试，说明如何通过命令行或 YAML 选择场景。

## Capabilities

### New Capabilities

- `deepsense6g-scene-selection`: 覆盖 DeepSense6G 场景选择、场景默认数据路径、场景 metadata 和新增场景的扩展约定。

### Modified Capabilities

- `modality-aware-data-loading`: dataset 构建必须根据场景选择解析 DeepSense6G 数据根目录和 split CSV，并在样本 metadata 中记录场景。
- `experiment-workflow`: 训练、评估和默认配置必须支持场景选择，并默认按场景归类输出运行目录。
- `experiment-artifact-registry`: 最佳 checkpoint registry 与 KD teacher 默认解析必须按场景隔离，避免不同场景同名 run 互相覆盖或误加载。

## Impact

- 主要影响 `src/kd_sensing/config/defaults.py`、`src/kd_sensing/config/io.py`、`src/kd_sensing/engine/builders.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/utils/artifact_registry.py`、`src/kd_sensing/data/datasets/scenario9.py` 和配置 YAML。
- 需要新增或重命名一个通用 DeepSense6G dataset 场景解析层，同时保留 `scenario9` 注册名兼容旧配置。
- 需要批量调整 canonical YAML 中的默认场景、`output.dir`、`paths.weights_dir`、registry 默认目录或等价解析逻辑，使新训练默认归入 `outputs/scene32/`，同时把已有训练产物迁移到 `outputs/scene9/`。
- 需要补充单元测试覆盖场景别名、默认路径解析、输出目录归类、resume/overwrite 行为、registry 隔离和 KD teacher fallback。
- 需要用 `conda run -n kd_mm_beam ...` 运行相关测试。
