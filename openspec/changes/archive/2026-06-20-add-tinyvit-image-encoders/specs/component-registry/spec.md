## MODIFIED Requirements

### Requirement: 默认组件导入包含新增组件
默认组件导入流程 MUST 注册 ResNet-18 image encoder、TinyViT image encoder、模块化序列模型和内置 core/head 组件，同时保持 registry 本身轻量可导入。导入 `kd_sensing.registries` MUST 不急切导入 torchvision、timm、dataset、训练器、checkpoint 文件、预训练权重接口或触发任何权重下载。

#### Scenario: 构建前导入默认组件
- **WHEN** 构建流程调用默认组件导入函数后再构建模块化序列模型
- **THEN** `MODELS` 注册表或对应子组件 registry MUST 包含新增注册名
- **AND** 用户配置中的新增注册名 MUST 可解析

#### Scenario: 轻量导入 registry 不触发 torchvision 或 TinyViT 权重
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import torchvision、timm 或 TinyViT 预训练权重接口
- **AND** 系统 MUST 不访问网络、不创建 checkpoint cache、不加载 TinyViT 权重

#### Scenario: 构建 TinyViT encoder 注册名
- **WHEN** 构建流程调用默认组件导入函数后查看 `ENCODERS.list()`
- **THEN** 输出 MUST 包含 `tinyvit_5m_scratch_rgb`、`tinyvit_5m_22k_rgb`、`tinyvit_11m_scratch_rgb` 和 `tinyvit_11m_22k_rgb`
- **AND** 系统 MUST 能通过 `ENCODERS.build()` 构建这些 TinyViT image encoder

#### Scenario: 未知 TinyViT 名称使用 registry 错误风格
- **WHEN** 用户请求不存在或拼写错误的 TinyViT encoder 注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称、registry 名称和可用 encoder 名称
