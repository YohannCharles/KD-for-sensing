## ADDED Requirements

### Requirement: JEPA downstream pooler 和 adapter 注册
项目 MUST 通过轻量组件构建边界暴露 JEPA downstream pooler 和 adapter。内置 mean pooler、GPS-query attention pooler 和 identity adapter MUST 能通过配置名称构建；未知 pooler 或 adapter 名称 MUST 使用现有 registry 错误风格报告。

#### Scenario: 按名称构建 mean pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `mean`
- **THEN** 系统 MUST 构建 mean pooler
- **AND** 该 pooler MUST 接收 patch tokens `[B,T,N,D]` 并输出 `[B,T,D]`

#### Scenario: 按名称构建 GPS-query pooler
- **WHEN** `jepa_context_image` 配置声明 downstream pooler 为 `gps_query_attention`
- **THEN** 系统 MUST 构建 GPS-query attention pooler
- **AND** 构建参数 MUST 支持 `k_queries`、`num_heads`、`condition_dim`、`latent_dim`、dropout 和 condition source

#### Scenario: 按名称构建 identity adapter
- **WHEN** `jepa_context_image` 配置未声明 adapter 或声明 adapter 为 `identity`
- **THEN** 系统 MUST 构建不改变输入 shape 的 identity adapter 或等价无操作路径
- **AND** 现有配置 MUST 无需新增 adapter 字段即可运行

#### Scenario: 未知 JEPA downstream 组件可诊断
- **WHEN** 用户配置不存在的 JEPA downstream pooler 或 adapter 名称
- **THEN** 系统 MUST 拒绝构建
- **AND** 错误信息 MUST 包含请求名称、组件类别和可用名称

### Requirement: JEPA downstream 注册保持轻量导入
JEPA downstream pooler/adapter 的注册 MUST 不破坏 registry 轻量导入边界。导入 `kd_sensing.registries` MUST 不 eager import torch model implementation、dataset、diagnostics、训练器或 checkpoint 文件；默认组件导入流程 MUST 显式注册内置 JEPA downstream 组件。

#### Scenario: 轻量导入 registry 不触发 JEPA model
- **WHEN** 开发者仅执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不 eager import `kd_sensing.models.jepa` 或 JEPA downstream 实现模块

#### Scenario: 默认组件导入后可构建 JEPA downstream 组件
- **WHEN** 构建流程调用默认组件导入函数
- **THEN** 内置 JEPA downstream pooler 和 adapter MUST 完成注册
- **AND** 用户配置中的内置 pooler/adapter 名称 MUST 可解析
