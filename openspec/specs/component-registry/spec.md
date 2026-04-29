# component-registry Specification

## Purpose
TBD - created by archiving change reorganize-project-structure. Update Purpose after archive.
## Requirements
### Requirement: 组件注册表
项目 MUST 提供轻量组件注册表，用于注册和构建模型、数据集、损失函数、指标、蒸馏器和预处理器。注册表 MUST 支持按字符串名称查询组件，并通过配置参数实例化组件。

#### Scenario: 按名称构建模型
- **WHEN** 配置中指定一个已注册模型名称和初始化参数
- **THEN** 系统 MUST 返回对应模型实例，并将配置参数传入模型构造函数

#### Scenario: 按名称构建数据集
- **WHEN** 配置中指定一个已注册数据集名称和初始化参数
- **THEN** 系统 MUST 返回对应 dataset 实例，并能被 DataLoader 使用

### Requirement: 可扩展模型和模态
新增 teacher、student、backbone、head、radar 或 fusion 模型时，开发者 MUST 能通过新增模块和注册名称扩展系统，而不需要复制训练脚本或修改训练循环主体。

#### Scenario: 新增 image-only student
- **WHEN** 开发者实现并注册一个新的 image-only student 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用现有 image-only 训练流程

#### Scenario: 新增多模态 fusion 模型
- **WHEN** 开发者实现并注册一个新的 image+radar fusion 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用现有 fusion 训练流程

#### Scenario: 新增 radar-only teacher
- **WHEN** 开发者实现并注册一个新的 radar-only teacher 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用 radar-only 训练和评估流程
- **AND** 模型 MUST 保持统一的 `(pred, features, output_features)` 输出约定

#### Scenario: 新增 radar-only student
- **WHEN** 开发者实现并注册一个新的 radar-only student 模型
- **THEN** 用户 MUST 能在配置中选择该模型，并复用 radar-only 训练、评估和 KD 流程
- **AND** 模型 MUST 保持统一的 `(pred, features, output_features)` 输出约定

### Requirement: 可扩展蒸馏方法
新增 KD 方法时，开发者 MUST 能通过新增 distiller 或 loss 组件并注册名称扩展系统。训练流程 MUST 将 student logits/features、teacher logits/features 和 labels 传入蒸馏组件，避免在训练循环内硬编码每种 KD 算法。

#### Scenario: 选择 logits KD
- **WHEN** 配置中选择 logits KD
- **THEN** 系统 MUST 构建对应蒸馏组件，并使用 temperature 和 alpha 参数计算蒸馏损失

#### Scenario: 选择 relational KD
- **WHEN** 配置中选择 relational KD
- **THEN** 系统 MUST 构建对应蒸馏组件，并使用距离权重、角度权重和 pair 采样参数计算关系蒸馏损失

#### Scenario: 新增蒸馏方法
- **WHEN** 开发者新增并注册一个蒸馏方法
- **THEN** 用户 MUST 能通过配置中的名称启用它，而不需要修改统一训练循环主体

### Requirement: 注册错误可诊断
注册表 MUST 对未知组件名称、重复注册名称和缺失必需参数提供明确错误信息，错误信息 MUST 包含注册表名称、请求的组件名称和可用组件列表或缺失字段。

#### Scenario: 请求未知组件
- **WHEN** 配置中引用未注册的模型、数据集、loss、metric、distiller 或 preprocessor 名称
- **THEN** 系统 MUST 抛出明确异常，并列出该注册表当前可用名称

#### Scenario: 重复注册组件名称
- **WHEN** 两个组件尝试注册到同一个注册表的相同名称
- **THEN** 系统 MUST 拒绝重复注册，并提示冲突名称和注册表类型

### Requirement: 组件发现文档
项目 MUST 在文档中说明如何查看可用组件、如何新增组件、如何在配置中引用组件，以及新增组件需要满足的输入输出约定。

#### Scenario: 按文档新增 metric
- **WHEN** 开发者按照 README 或扩展指南新增并注册一个 metric
- **THEN** 该 metric MUST 能被评估配置引用，并出现在评估结果输出中

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

