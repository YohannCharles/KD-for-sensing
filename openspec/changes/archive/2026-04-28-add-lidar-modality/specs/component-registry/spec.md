## ADDED Requirements

### Requirement: LiDAR 组件注册
项目 MUST 通过现有组件注册表注册 LiDAR 模型和预处理器，使用户能通过配置构建 LiDAR teacher、student、feature extractor、dataset 处理路径和离线预处理流程。

#### Scenario: 按名称构建 LiDAR teacher
- **WHEN** 配置中指定 `type: lidar_teacher` 及其初始化参数
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 `LidarModalityNet` 实例

#### Scenario: 按名称构建 LiDAR student
- **WHEN** 配置中指定 `type: lidar_student` 及其初始化参数
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 `LidarStudentModalityNet` 实例

#### Scenario: 按名称构建 LiDAR 预处理器
- **WHEN** 配置中指定 LiDAR BEV 预处理器名称及其初始化参数
- **THEN** 系统 MUST 通过 `PREPROCESSORS` 注册表返回可运行的 LiDAR 预处理器实例

### Requirement: LiDAR 注册错误可诊断
LiDAR 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 LiDAR 组件
- **WHEN** 配置中引用未注册的 LiDAR 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: LiDAR 构建参数缺失
- **WHEN** 配置中引用已注册 LiDAR 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误
