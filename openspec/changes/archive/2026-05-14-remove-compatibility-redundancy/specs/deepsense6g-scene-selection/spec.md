## MODIFIED Requirements

### Requirement: DeepSense6G 场景选择配置
项目 MUST 支持通过配置选择 DeepSense6G 场景。`data.dataset.type` MUST 使用 `deepsense6g`，`data.dataset.scene` MUST 接受整数和字符串别名，首批 MUST 支持 Scenario 9 与 Scenario 32。未显式设置场景时，通用 DeepSense6G 配置 MUST 默认使用 Scenario 32。旧 `the scene-9 dataset-type spelling`、`scenario31` 和 `scenario32` 配置 MUST 被拒绝并给出迁移提示。

#### Scenario: 默认使用 Scenario 32
- **WHEN** 用户运行未显式设置 `data.dataset.scene` 的默认 DeepSense6G 训练配置
- **THEN** 系统 MUST 将场景解析为 Scenario 32
- **AND** 默认数据根目录 MUST 指向 Scenario 32 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 32` 和 `scene_slug: scene32`

#### Scenario: 通过整数选择 Scenario 9
- **WHEN** 用户设置 `data.dataset.type: deepsense6g` 且 `data.dataset.scene: 9`
- **THEN** 系统 MUST 将场景解析为 Scenario 9
- **AND** 默认数据根目录 MUST 指向 Scenario 9 的数据目录
- **AND** 运行 metadata MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 通过别名选择场景
- **WHEN** 用户设置 `data.dataset.scene` 为 `scene9`、`scenario9`、`scene32` 或 `scenario32`
- **THEN** 系统 MUST 解析到对应的规范场景编号
- **AND** 配置中的大小写差异 MUST 不影响解析结果

#### Scenario: 旧 dataset type 被拒绝
- **WHEN** 用户设置 `the scene-9 dataset-type spelling`、`scenario31` 或 `scenario32`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和对应 `data.dataset.scene`

#### Scenario: 未知场景被拒绝
- **WHEN** 用户设置未注册的 `data.dataset.scene`
- **THEN** 系统 MUST 拒绝构建配置或 dataset
- **AND** 错误信息 MUST 列出当前支持的场景
