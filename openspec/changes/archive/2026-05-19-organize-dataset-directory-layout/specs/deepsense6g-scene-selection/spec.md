## MODIFIED Requirements

### Requirement: 场景默认路径和显式覆盖
DeepSense6G 场景解析 MUST 为每个支持场景提供默认数据根目录、legacy 数据根目录、train CSV 名和 test CSV 名。未显式配置 `data_root` 时，默认数据根目录 MUST 使用 `dataset/DeepSense6G/scenario*` 家族目录。用户显式配置的 `data_root`、`train_csv_name` 或 `test_csv_name` MUST 覆盖场景默认值。

#### Scenario: 使用默认场景路径
- **WHEN** 用户选择 Scenario 31 或未显式设置 `data.dataset.scene`，且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 `dataset/DeepSense6G/scenario31` 作为 Scenario 31 的默认数据根目录
- **AND** train/test dataset MUST 使用该场景的默认 split CSV 名

#### Scenario: 显式 data_root 覆盖默认值
- **WHEN** 用户选择 Scenario 31 并设置 `data.dataset.data_root: /tmp/custom_scene31`
- **THEN** 系统 MUST 使用 `/tmp/custom_scene31` 构建 dataset
- **AND** 系统 MUST 仍在 metadata 中记录规范场景为 Scenario 31

#### Scenario: 显式旧 data_root 兼容
- **WHEN** 用户选择 Scenario 31 并设置 `data.dataset.data_root: dataset/scenario31`
- **THEN** 系统 MUST 使用 `dataset/scenario31` 构建 dataset
- **AND** 系统 MUST 不把该显式路径改写为 `dataset/DeepSense6G/scenario31`

#### Scenario: 显式 CSV 覆盖默认值
- **WHEN** 用户设置 `data.dataset.train_csv_name` 或 `data.dataset.test_csv_name`
- **THEN** 系统 MUST 使用显式 CSV 名构建对应 split
- **AND** 场景默认 CSV 名不得覆盖用户显式设置

## ADDED Requirements

### Requirement: DeepSense6G 规范目录清单
DeepSense6G MUST 将首批支持场景的规范数据根目录定义为 `dataset/DeepSense6G/scenario9`、`dataset/DeepSense6G/scenario31` 和 `dataset/DeepSense6G/scenario32`。这些默认路径 MUST 由同一场景描述符或 dataset layout descriptor 提供。

#### Scenario: Scenario 9 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 9`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario9`

#### Scenario: Scenario 31 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 31`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario31`

#### Scenario: Scenario 32 规范路径
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`，并且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 将默认数据根目录解析为 `dataset/DeepSense6G/scenario32`
