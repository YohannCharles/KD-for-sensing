## ADDED Requirements

### Requirement: 时序缺失配置契约保持轻量导入
项目 MUST 将时序缺失模式、时序聚合方式及其纯配置规范化与 tensor/mask runtime 实现解耦。配置加载、配置 normalization 和 configuration validation 路径 MUST 能解析并校验这些字段，且不得因此导入 `torch`、模型实现、dataset runtime、诊断渲染或训练主循环。依赖方向 MUST 只允许 tensor runtime 指向该纯配置契约，纯配置契约 MUST NOT 反向依赖 tensor runtime；现有时序缺失配置值、错误语义和运行行为 MUST 保持兼容。

#### Scenario: 冷进程导入配置包不加载 tensor runtime
- **WHEN** 开发者在 fresh Python process 中执行 `import kd_sensing.config`
- **THEN** 导入 MUST 成功
- **AND** `torch`、时序缺失 tensor runtime、模型实现、dataset runtime、诊断渲染和训练主循环 MUST 不出现在已加载模块中

#### Scenario: 配置路径独立规范化时序字段
- **WHEN** 配置 normalization 或 validation 解析合法或非法的时序缺失模式与时序聚合方式
- **THEN** 系统 MUST 使用单一纯配置契约返回规范化值或清晰拒绝非法值
- **AND** 该过程 MUST 不要求导入 `torch` 或构建 tensor、dataset、model

#### Scenario: 时序 tensor runtime 行为保持兼容
- **WHEN** 训练、评估或 difficulty operator 调用现有时序聚合、mask 采样或 batch 变换函数
- **THEN** 这些函数 MUST 继续使用相同的模式名称、默认值、错误语义和 tensor 行为
- **AND** 现有调用方 MUST 不需要通过新的兼容 facade 或 package-level 聚合层访问这些函数
