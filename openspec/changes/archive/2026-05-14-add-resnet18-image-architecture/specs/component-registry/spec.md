## ADDED Requirements

### Requirement: 模块化模型组件注册
项目 MUST 通过现有组件注册边界暴露新的模块化序列模型及其可复用子组件。新增 image encoder、projector、representation core 和 head MUST 能通过配置名称构建，且不得要求训练脚本手写实例化逻辑。

#### Scenario: 按名称构建模块化序列模型
- **WHEN** 配置指定新的模块化序列模型注册名及其子组件配置
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建模型
- **AND** 构建参数 MUST 来自配置字段
- **AND** 训练循环 MUST 不需要为该注册名新增专用 forward 分支

#### Scenario: 按名称构建 ResNet-18 image encoder
- **WHEN** 模块化模型配置选择 `resnet18_imagenet_rgb` image encoder
- **THEN** 系统 MUST 通过注册表或明确 factory 构建该 encoder
- **AND** 未知 encoder 名称 MUST 使用现有 registry 错误风格报告可用名称

### Requirement: 默认组件导入包含新增组件
默认组件导入流程 MUST 注册 ResNet-18 image encoder、模块化序列模型和内置 core/head 组件，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 torchvision、dataset、训练器或 checkpoint 文件。

#### Scenario: 构建前导入默认组件
- **WHEN** 构建流程调用默认组件导入函数后再构建模块化序列模型
- **THEN** `MODELS` 注册表或对应子组件 registry MUST 包含新增注册名
- **AND** 用户配置中的新增注册名 MUST 可解析

#### Scenario: 轻量导入 registry 不触发 torchvision
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import torchvision 或 ResNet-18 预训练权重接口

### Requirement: 模块化组件错误可诊断
模块化模型构建失败时，系统 MUST 抛出包含组件类别、请求名称、相关模态和可用名称的清晰错误。shape 或 profile 不匹配错误 MUST 在构建或首次 forward 的早期暴露，并包含实际输入 shape。

#### Scenario: 未知 representation core
- **WHEN** 用户配置不存在的 `representation_core.type`
- **THEN** 系统 MUST 拒绝构建模块化序列模型
- **AND** 错误信息 MUST 包含请求的 core 名称和可用 core 名称

#### Scenario: encoder 与 profile 不匹配
- **WHEN** 用户配置 `rgb_imagenet` profile 但 image encoder 只支持 1 通道输入
- **THEN** 系统 MUST 拒绝构建或首次 forward
- **AND** 错误信息 MUST 包含 image profile、encoder 名称、期望通道数和实际通道数
