## ADDED Requirements

### Requirement: 默认组件延迟导入
组件注册系统 MUST 保持注册表本身轻量可导入。导入 `kd_sensing.registries` MUST 不自动导入默认 dataset、model、preprocessor、diagnostics 或训练模块；默认组件注册 MUST 由显式注册导入函数或构建流程触发。

#### Scenario: 轻量导入 registry
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入默认 dataset、model 或 preprocessor 模块

#### Scenario: 构建前导入默认组件
- **WHEN** 构建流程需要通过 registry 构建已内置的 dataset、model、loss、metric、distiller 或 preprocessor
- **THEN** 构建流程 MUST 在查询 registry 前触发默认组件导入
- **AND** 已有配置中的 registry type MUST 继续可解析

### Requirement: 包级导出不扩大依赖面
包级 `__init__.py` 文件 MUST 避免 eager re-export 会引入重依赖或默认组件注册的符号。需要重依赖的功能 MUST 通过窄模块路径导入，或通过明确的延迟导入机制暴露。

#### Scenario: 导入 utils 包不触发 artifact registry
- **WHEN** 开发者执行 `import kd_sensing.utils`
- **THEN** 导入 MUST 不要求 dataset 场景、checkpoint registry 或 torch checkpoint 相关模块完成导入
- **AND** 路径和 seed 等轻量工具 MUST 仍可通过窄路径导入

#### Scenario: 显式导入 artifact registry
- **WHEN** 训练或评估代码需要 checkpoint registry 功能
- **THEN** 代码 MUST 从 `kd_sensing.utils.artifact_registry` 或等价窄入口导入
- **AND** checkpoint registry 行为 MUST 与变更前保持兼容

### Requirement: 注册发现文档区分轻量导入与组件注册
扩展文档 MUST 说明 registry 对象导入和默认组件注册是两个不同动作。文档 MUST 指导开发者在查看内置组件列表前显式导入默认组件或对应组件模块。

#### Scenario: 按文档查看内置模型
- **WHEN** 开发者按照扩展文档查看 `MODELS.list()`
- **THEN** 文档 MUST 要求先触发默认模型模块导入或调用默认组件导入函数
- **AND** 输出 MUST 包含内置模型注册名

#### Scenario: 按文档注册自定义组件
- **WHEN** 开发者在自定义模块中注册一个新组件
- **THEN** 文档 MUST 说明该模块需要在构建前被导入
- **AND** 系统 MUST 不通过扫描整个仓库隐式导入未知模块
