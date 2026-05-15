## ADDED Requirements

### Requirement: DeepSense6G target provider
DeepSense6G dataset MUST 通过 target provider 组件构造 beam、occlusion、position 和 multitask 所需 target 字段。dataset 主类 MAY 协调 provider 的生命周期，但新增 target 类型 MUST 不要求修改主类的核心 `__getitem__` 取样流程。

#### Scenario: occlusion target 由 provider 生成
- **WHEN** dataset 配置启用 `occlusion_target`
- **THEN** occlusion target provider MUST 读取所需 mmWave power 或统计信息并生成 `occlusion_label` 与 `occlusion_valid`
- **AND** 返回样本字段、dtype 和 shape MUST 与现有 occlusion objective 训练兼容

#### Scenario: position target 由 provider 生成
- **WHEN** dataset 配置启用 `position_target`
- **THEN** position target provider MUST 读取或构造位置 target 并生成 `position_target` 与 `position_valid`
- **AND** provider MUST 复用既有 normalization/scaler 语义

#### Scenario: 未启用 target 不读取相关资源
- **WHEN** 当前 objective 不需要 occlusion 或 position target
- **THEN** 对应 target provider MUST 不读取 mmWave power、GPS future position 或 position scaler 资源
- **AND** 返回样本 MUST 不包含未启用 target 字段

### Requirement: DeepSense6G 模态 loader 组件
DeepSense6G dataset MUST 将 image、radar、GPS、LiDAR 和 mmWave 的文件读取、cache 访问和特征构造委托给模态 loader 组件。未启用模态的 loader MUST 不初始化重资源，也不得读取该模态文件。

#### Scenario: GPS+mmWave fusion 只初始化相关 loader
- **WHEN** 配置启用 fusion modalities `["gps", "mmwave"]`
- **THEN** dataset MUST 只初始化 GPS、mmWave、beam label 和启用 target 所需组件
- **AND** dataset MUST 不初始化 image、radar 或 LiDAR loader 的重资源

#### Scenario: 新增模态 loader 不影响 target provider
- **WHEN** 开发者新增或修改某个输入模态 loader
- **THEN** 变更 MUST 不要求编辑 occlusion 或 position target provider
- **AND** target provider 测试 MUST 继续通过
