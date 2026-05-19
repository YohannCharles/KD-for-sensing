## ADDED Requirements

### Requirement: CSI 组件注册
项目 MUST 通过现有组件注册表注册 CSI encoder 和可选 CSI 模型入口，使用户能通过配置构建 pilot dual-view CSI encoder，并复用现有 `modular_sequence` 训练流程。

#### Scenario: 按名称构建 CSI encoder
- **WHEN** 配置中指定 `type: pilot_dual_view_csi` 及其初始化参数
- **THEN** 系统 MUST 通过 `ENCODERS` 注册表返回 CSI encoder 实例
- **AND** 构建参数 MUST 支持 `output_dim`、`d_model`、pilot estimation、dual-view、tokenizer、temporal 和 dropout 相关字段

#### Scenario: 默认组件导入包含 CSI 模块
- **WHEN** 构建流程调用默认组件导入函数后再构建 `pilot_dual_view_csi`
- **THEN** `ENCODERS` 注册表 MUST 包含 `pilot_dual_view_csi`
- **AND** 注册表轻量导入边界 MUST 与现有 registry 语义一致

### Requirement: CSI 注册错误可诊断
CSI 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 CSI encoder
- **WHEN** 配置中引用未注册的 CSI encoder 名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: CSI 构建参数非法
- **WHEN** 配置中引用 `pilot_dual_view_csi` 但提供非法 `view_fusion` 或非正数 `pilot_len`
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含非法字段或原始构建错误
