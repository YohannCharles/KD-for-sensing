## Why

当前训练默认仍指向 Scenario 32，默认配置、KD checkpoint 路径、输出目录和文档中的默认说明都围绕 `scene32` 展开。将默认训练场景切换到 Scenario 31 后，未显式覆盖场景的训练流程应直接使用新的目标数据集，并避免继续读写 Scenario 32 的默认产物。

## What Changes

- 将未显式设置 `data.dataset.scene` 的 DeepSense6G 训练默认解析为 Scenario 31。
- 将默认训练配置中的 `data.dataset.scene`、`scene_id`、`scene_slug` 和默认数据根目录切到 `31`、`scene31`、`dataset/scenario31`。
- 将默认训练输出、resume、最佳 checkpoint registry 和 teacher reliability registry 的默认路径切到 `outputs/scene31/...`。
- 将 KD、fusion、canonical 和脚本中依赖默认 teacher checkpoint 或 registry 的相对路径从 `scene32` 切到 `scene31`，保留显式覆盖其它场景的能力。
- 更新相关 README 和测试断言，明确 Scenario 9、Scenario 31、Scenario 32 都可通过 `data.dataset.scene` 显式选择。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `deepsense6g-scene-selection`: 默认 DeepSense6G 场景从 Scenario 32 改为 Scenario 31，并将 Scenario 31 纳入受支持的显式场景别名。
- `modality-aware-data-loading`: 通用 `deepsense6g` 数据构建需要支持 Scenario 31 默认路径与 split CSV 解析。
- `experiment-workflow`: 默认训练、resume 和运行输出分组从 `outputs/scene32` 改为 `outputs/scene31`。
- `experiment-artifact-registry`: 默认 checkpoint registry 和 teacher reliability registry 路径应随默认场景切到 `scene31`。

## Impact

- 影响默认配置与 canonical 配置解析：`src/kd_sensing/config/defaults.py`、`src/kd_sensing/config/canonical.py`。
- 影响场景描述符和别名解析：`src/kd_sensing/data/scenes.py`。
- 影响训练、评估、teacher registry 和分析脚本的默认路径。
- 影响 `configs/**` 中默认 `scene: 32`、`outputs/scene32`、`dataset/scenario32` 的训练相关配置。
- 影响 README 和测试中关于默认场景、默认输出目录、默认 checkpoint 路径的断言。
