## MODIFIED Requirements

### Requirement: Registry 只暴露 canonical 入口
组件注册表 MUST 只注册当前 canonical dataset、model、loss、metric、distiller 和 preprocessor 名称。已经由 canonical 名称替代的场景专用 dataset alias、旧模型类名 alias、legacy encoder alias，以及已退役的 CRAF、MARF、G2D 和 Multimodal-NF 入口 MUST 不再注册。

#### Scenario: dataset registry 不含场景专用 alias
- **WHEN** 构建流程导入默认 dataset 组件
- **THEN** `DATASETS` MUST 包含 `deepsense6g`
- **AND** `DATASETS` MUST 不包含 `scenario9`、`scenario31` 或 `scenario32`

#### Scenario: 旧 dataset alias 构建失败
- **WHEN** 用户配置 `the scene-9 dataset-type spelling`
- **THEN** registry 构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 指向 `data.dataset.type: deepsense6g` 和 `data.dataset.scene: 9`

#### Scenario: 旧研究线入口构建失败
- **WHEN** 用户请求构建 `craf_fusion`、`marf_fusion`、`distillation.type: g2d` 或 `data.dataset.type: multimodal_nf`
- **THEN** registry 或配置构建 MUST 拒绝该名称
- **AND** 系统 MUST 不通过 deprecated alias、overlay 或兼容 facade 重定向到其它实现

#### Scenario: 旧模型类名 alias 不再导出
- **WHEN** 开发者从模型模块导入旧 fusion 类名 alias
- **THEN** 导入 MUST 失败或触发清晰迁移错误
- **AND** 错误信息 MUST 指向当前公开类名或 canonical 模型注册名

### Requirement: 已删除组件错误可诊断
当用户引用已删除的兼容组件名称或退役研究线组件名称时，注册表错误 MUST 区分“未知名称”和“已删除名称”。已删除名称的错误信息 MUST 包含当前支持范围或迁移方向。

#### Scenario: 已删除 dataset type
- **WHEN** 用户请求构建 `scenario9` dataset
- **THEN** 系统 MUST 抛出包含 `scenario9` 的错误
- **AND** 错误信息 MUST 说明该名称已删除并给出 `deepsense6g + scene` 配置示例

#### Scenario: 已删除模型 alias
- **WHEN** 用户请求构建旧 fusion 类名 alias 或已删除 image encoder alias
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 列出当前支持的 canonical 注册名

#### Scenario: 已退役研究线组件
- **WHEN** 用户请求 `craf_fusion`、`marf_fusion`、`g2d` distiller 或 `multimodal_nf` dataset
- **THEN** 系统 MUST 抛出包含请求名称的错误
- **AND** 错误信息 MUST 说明该入口已退役且不提供兼容迁移

## REMOVED Requirements

### Requirement: CRAF 组件注册
**Reason**: CRAF 架构退役，不再通过组件注册表暴露 CRAF 模型、baseline 或 loss。
**Migration**: 使用当前保留的 canonical fusion 模型和训练配置。

#### Scenario: CRAF 注册名退役
- **WHEN** 配置指定 `type: craf_fusion`
- **THEN** 系统 MUST 不再通过 `MODELS` 注册表构建 CRAF 模型
- **AND** registry 错误 MUST 指出该名称不可用

### Requirement: 默认组件导入包含 CRAF
**Reason**: 默认组件导入不应注册已退役 CRAF 组件。
**Migration**: 默认组件导入只注册当前保留的模型、dataset、loss、metric、distiller 和 preprocessor。

#### Scenario: 默认导入不含 CRAF
- **WHEN** 构建流程调用默认组件导入函数
- **THEN** `MODELS` 注册表 MUST 不包含 `craf_fusion`
- **AND** 导入流程 MUST 不导入 CRAF 模型模块

### Requirement: CRAF loss helper 可测试
**Reason**: CRAF loss helper 只服务于已退役 CRAF 训练路径。
**Migration**: 删除 CRAF loss helper 正向测试；保留当前通用 loss 和 distiller 测试。

#### Scenario: CRAF loss helper 测试退役
- **WHEN** 开发者运行 focused tests
- **THEN** 测试 MUST 不再要求 beam soft、gate target 或 context marginal CRAF helper 存在

### Requirement: Teacher-prior CRAF 组件注册
**Reason**: teacher-prior CRAF 组件、gate、prior 和 KD loss 随 CRAF 退役。
**Migration**: 不提供兼容组件；新增 gate 或 prior 方法需重新提出 capability。

#### Scenario: teacher-prior CRAF 组件不可用
- **WHEN** 配置选择 teacher-prior CRAF 所需模型、gate 或 loss
- **THEN** 系统 MUST 拒绝该组件
- **AND** 默认组件导入 MUST 不注册 teacher-prior CRAF 组件

### Requirement: Teacher loader 组件边界
**Reason**: 该 teacher loader 组件专用于 teacher-prior CRAF/MARF encoder 初始化。
**Migration**: 普通 checkpoint 加载继续由训练/评估 workflow 处理。

#### Scenario: teacher loader 退役
- **WHEN** 测试尝试直接调用 teacher-prior CRAF/MARF teacher loader
- **THEN** 系统 MUST 不要求提供该 loader
- **AND** 对应测试 MUST 被删除

### Requirement: 默认导入保持轻量
**Reason**: 该要求专门约束新增 teacher-prior CRAF 组件的默认导入；组件已退役。
**Migration**: 默认导入轻量边界继续由通用 registry 要求约束。

#### Scenario: CRAF 默认导入要求删除
- **WHEN** 开发者导入 `kd_sensing.registries`
- **THEN** 系统 MUST 不导入 teacher-prior CRAF 组件
- **AND** active specs MUST 不再要求构建 CRAF 前导入默认组件
