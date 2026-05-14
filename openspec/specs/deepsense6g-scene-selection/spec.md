# deepsense6g-scene-selection Specification

## Purpose
TBD - created by archiving change support-deepsense6g-scene-selection. Update Purpose after archive.
## Requirements
### Requirement: DeepSense6G 场景选择配置
项目 MUST 支持通过配置选择 DeepSense6G 场景。`data.dataset.type` MUST 使用 `deepsense6g`，`data.dataset.scene` MUST 接受整数和字符串别名，首批 MUST 支持 Scenario 9、Scenario 31 与 Scenario 32。未显式设置场景时，通用 DeepSense6G 配置 MUST 默认使用 Scenario 31。旧 `the scene-9 dataset-type spelling`、`scenario31` 和 `scenario32` dataset type 配置 MUST 被拒绝并给出迁移提示。

#### Scenario: 默认使用 Scenario 31
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 训练配置
- **THEN** 系统 MUST 将场景解析为 Scenario 31
- **AND** 默认数据根目录 MUST 指向 Scenario 31 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 31` 和 `scene_slug: scene31`

#### Scenario: 通过整数选择 Scenario 9
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 9`
- **THEN** 系统 MUST 将场景解析为 Scenario 9
- **AND** 默认数据根目录 MUST 指向 Scenario 9 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 通过整数选择 Scenario 32
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 32`
- **THEN** 系统 MUST 将场景解析为 Scenario 32
- **AND** 默认数据根目录 MUST 指向 Scenario 32 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 32` 和 `scene_slug: scene32`

#### Scenario: 通过别名选择场景
- **WHEN** 用户设置 `data.dataset.scene` 为 `scene9`、`scenario9`、`scene31`、`scenario31`、`scene32` 或 `scenario32`
- **THEN** 系统 MUST 解析到对应的规范场景编号
- **AND** 配置中的大小写差异 MUST 不影响解析结果

#### Scenario: 旧 dataset type 被拒绝
- **WHEN** 用户设置 `the scene-9 dataset-type spelling`、`scenario31` 或 `scenario32` 作为 `data.dataset.type`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和对应 `data.dataset.scene`

#### Scenario: 未知场景被拒绝
- **WHEN** 用户设置未注册的 `data.dataset.scene`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 列出当前支持的场景

### Requirement: 场景默认路径和显式覆盖
DeepSense6G 场景解析 MUST 为每个支持场景提供默认数据根目录、train CSV 名和 test CSV 名。用户显式配置的 `data_root`、`train_csv_name` 或 `test_csv_name` MUST 覆盖场景默认值。

#### Scenario: 使用默认场景路径
- **WHEN** 用户选择 Scenario 31 或未显式设置 `data.dataset.scene`，且未显式设置 `data.dataset.data_root`
- **THEN** 系统 MUST 使用 Scenario 31 的默认数据根目录
- **AND** train/test dataset MUST 使用该场景的默认 split CSV 名

#### Scenario: 显式 data_root 覆盖默认值
- **WHEN** 用户选择 Scenario 31 并设置 `data.dataset.data_root: /tmp/custom_scene31`
- **THEN** 系统 MUST 使用 `/tmp/custom_scene31` 构建 dataset
- **AND** 系统 MUST 仍在 metadata 中记录规范场景为 Scenario 31

#### Scenario: 显式 CSV 覆盖默认值
- **WHEN** 用户设置 `data.dataset.train_csv_name` 或 `data.dataset.test_csv_name`
- **THEN** 系统 MUST 使用显式 CSV 名构建对应 split
- **AND** 场景默认 CSV 名不得覆盖用户显式设置

### Requirement: 场景扩展约定
项目 MUST 提供可维护的场景注册或描述符机制，使未来新增 DeepSense6G 场景不需要复制数据集类或重写训练入口。

#### Scenario: 新增场景描述符
- **WHEN** 开发者新增一个包含场景编号、slug、别名和默认路径的场景描述符
- **THEN** 配置解析、dataset 构建、输出目录分组和 metadata 记录 MUST 能复用同一描述符
- **AND** 新增场景不得要求复制 Scenario 9 数据集读取逻辑

