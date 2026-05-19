## Why

当前 DeepSense6G 的场景目录直接散落在 `dataset/scenario31`、`dataset/scenario32`、`dataset/scenario9`，路径规则分布在场景注册、预处理配置、README 和测试中。后续还可能加入 MMW 的 sunny/rainy/foggy 数据，每个天气下再区分 `Sensor_Data` 与 `Channel_Data`，如果继续按场景或数据集临时写路径，会使数据管理、迁移和配置复用变得困难。

## What Changes

- 建立统一的数据集家族目录约定：DeepSense6G 场景统一放入 `dataset/DeepSense6G/scenario31`、`dataset/DeepSense6G/scenario32`、`dataset/DeepSense6G/scenario9`。
- 保持 `data.dataset.type: deepsense6g` 与 `data.dataset.scene` 的现有使用方式，默认解析到新的 DeepSense6G 家族目录。
- 保留旧 `dataset/scenario*` 路径的显式兼容：用户显式配置旧 `data_root` 时继续可用；未显式配置时只使用新的规范默认目录。
- 抽出数据集目录 layout/descriptor 层，集中维护数据集家族、场景/条件、默认根目录、默认 CSV 名称和别名，减少路径字符串散落。
- 为将来的 MMW 数据预留目录规范：`dataset/MMW/<sunny|rainy|foggy>/Sensor_Data` 保存传感器数据，`dataset/MMW/<sunny|rainy|foggy>/Channel_Data` 保存信道数据。
- 更新预处理配置、文档和测试，使 DeepSense6G 的默认路径、场景覆盖和序列 CSV 生成都指向新目录。
- 不改变 DeepSense6G 序列 CSV 的列名协议，也不改变 CSV 内模态文件相对路径的读取方式。

## Capabilities

### New Capabilities
- `dataset-directory-layout`: 定义项目级数据集家族目录规范、默认根目录解析、旧目录兼容原则，以及未来 MMW 天气/数据类型子目录约定。

### Modified Capabilities
- `deepsense6g-scene-selection`: DeepSense6G 场景默认数据根目录从 `dataset/scenario*` 调整为 `dataset/DeepSense6G/scenario*`，同时保留场景别名与显式 `data_root` 覆盖能力。
- `modality-aware-data-loading`: 数据加载和预处理流程必须通过集中 layout 解析 DeepSense6G 根目录，并继续以 scene root 为基准解析 CSV 内相对文件路径。

## Impact

- 影响 `src/kd_sensing/data/scenes.py` 的场景描述符和默认路径。
- 可能新增 `src/kd_sensing/data/layouts.py` 或等价模块，用于集中管理数据集家族目录。
- 影响 `src/kd_sensing/cli/preprocess.py`、`configs/preprocess/*.yaml`、README 中的默认数据路径。
- 影响场景路径相关测试，如 `tests/test_training_io_workflow.py`、`tests/test_modality_visual_diagnostics.py`。
- 不引入新的第三方依赖；不迁移或复制本地大数据文件，只更新代码和文档中的目录规范。
