## ADDED Requirements

### Requirement: Registry 只暴露 canonical 入口
组件注册表 MUST 只注册当前 canonical dataset、model、loss、metric、distiller 和 preprocessor 名称。已经由 canonical 名称替代的场景专用 dataset alias、旧模型类名 alias 和 legacy encoder alias MUST 不再注册。

#### Scenario: dataset registry 不含场景专用 alias
- **WHEN** 构建流程导入默认 dataset 组件
- **THEN** `DATASETS` MUST 包含 `deepsense6g`
- **AND** `DATASETS` MUST 不包含 `scenario9`、`scenario31` 或 `scenario32`

#### Scenario: 旧 dataset alias 构建失败
- **WHEN** 用户配置 `the scene-9 dataset-type spelling`
- **THEN** registry 构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 旧模型类名 alias 不再导出
- **WHEN** 开发者从模型模块导入旧 fusion 类名 alias
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向当前公开类名或 canonical 模型注册名

### Requirement: 默认组件导入不依赖兼容模块
默认组件导入流程 MUST 注册 canonical 内置组件，同时保持 registry 本身轻量可导入。默认组件导入 MUST 不通过 `scenario9.py`、`engine.builders`、`data.transforms` 或 `_legacy` 兼容模块完成。

#### Scenario: 导入默认 dataset 组件
- **WHEN** 构建流程调用默认组件导入函数后再构建 DeepSense6G dataset
- **THEN** 默认导入 MUST 加载场景中立 dataset 模块
- **AND** 系统 MUST 不导入 `kd_sensing.data.datasets.scenario9`

#### Scenario: registry 轻量导入
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 dataset、model、training、checkpoint 或兼容 facade 模块

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称时，注册表错误 MUST 区分“未知名称”和“已删除名称”。已删除名称的错误信息 MUST 包含迁移路径。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求构建旧 fusion 类名 alias 或已删除 image encoder alias
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名
