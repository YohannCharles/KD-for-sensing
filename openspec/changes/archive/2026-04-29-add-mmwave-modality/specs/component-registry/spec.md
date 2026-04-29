## ADDED Requirements

### Requirement: mmWave 组件注册
项目 MUST 通过现有组件注册表注册 mmWave 模型和预处理器，使用户能通过配置构建 mmWave teacher、student、feature extractor、dataset 处理路径和序列预处理流程。

#### Scenario: 按名称构建 mmWave teacher
- **WHEN** 配置中指定 `type: mmwave_teacher` 及其初始化参数
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 `MmWaveModalityNet` 实例

#### Scenario: 按名称构建 mmWave student
- **WHEN** 配置中指定 `type: mmwave_student` 及其初始化参数
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 `MmWaveStudentModalityNet` 实例

#### Scenario: 按名称构建 mmWave feature extractor
- **WHEN** 配置中指定 `type: mmwave_feature_extractor` 及其初始化参数
- **THEN** 系统 MUST 通过 `MODELS` 注册表返回 `MmWaveFeatureExtractor` 实例

#### Scenario: 按名称运行 mmWave 序列预处理
- **WHEN** 配置中指定序列预处理器并启用 `include_mmwave: true`
- **THEN** 系统 MUST 通过 `PREPROCESSORS` 注册表构建可运行的序列预处理器
- **AND** 预处理器 MUST 输出可被 mmWave dataset 路径读取的 `mmwave1..mmwaveN` 列

### Requirement: mmWave 注册错误可诊断
mmWave 相关注册错误 MUST 使用现有注册表错误风格，并在未知名称、重复名称或缺失必需参数时提供清晰错误信息。

#### Scenario: 请求未知 mmWave 组件
- **WHEN** 配置中引用未注册的 mmWave 模型或预处理器名称
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含请求的组件名称、注册表名称和可用组件列表

#### Scenario: mmWave 构建参数缺失
- **WHEN** 配置中引用已注册 mmWave 组件但缺少必需构造参数
- **THEN** 系统 MUST 抛出明确异常
- **AND** 错误信息 MUST 包含缺失字段或原始构建错误
