# component-registry Specification

## Purpose
Define the lightweight registry contract for models, datasets, losses, metrics, distillers, and preprocessors, including explicit default component import boundaries and extension guidance.
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

### Requirement: CRAF 组件注册
CRAF 相关模型和 loss 组件 MUST 通过现有组件注册或明确的窄模块入口接入系统。新增组件 MUST 能通过配置名称构建，并 MUST 不要求训练脚本手写实例化逻辑。

#### Scenario: 按名称构建 CRAF 模型
- **WHEN** 配置指定 `type: craf_fusion`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 CRAF 模型
- **AND** 构建参数 MUST 来自配置字段

#### Scenario: 按名称构建 token transformer baseline
- **WHEN** 配置指定 token transformer fusion baseline 的注册名
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建该 baseline

#### Scenario: 注册错误可诊断
- **WHEN** 用户引用不存在的 CRAF 组件注册名
- **THEN** 系统 MUST 使用现有 registry 错误风格抛出异常
- **AND** 错误信息 MUST 包含请求名称和可用组件列表

### Requirement: 默认组件导入包含 CRAF
默认组件导入流程 MUST 注册 CRAF 内置组件，同时保持 registry 轻量导入边界。

#### Scenario: 构建流程导入默认组件
- **WHEN** 构建流程调用默认组件导入函数后再构建 `craf_fusion`
- **THEN** `MODELS` 注册表 MUST 包含 CRAF 注册名

#### Scenario: 轻量导入 registry
- **WHEN** 开发者仅导入 `kd_sensing.registries`
- **THEN** 系统 MUST 不 eager import CRAF 模型依赖
- **AND** 轻量导入边界 MUST 与现有 registry 语义一致

### Requirement: CRAF loss helper 可测试
CRAF 使用的 beam soft loss、sequence CE/per-sample loss 和 gate supervision helper MUST 有明确模块边界，并 MUST 能被单元测试直接调用。

#### Scenario: 直接测试 beam soft loss
- **WHEN** 测试代码传入 logits、labels、beam 数量和 sigma
- **THEN** helper MUST 返回标量 loss
- **AND** ignore index 位置 MUST 不影响 loss

#### Scenario: 直接测试 gate target
- **WHEN** 测试代码传入 full loss 与 drop loss
- **THEN** helper MUST 返回范围可控的模态贡献目标
- **AND** 目标 MUST 能与 reliability gate 计算监督 loss

#### Scenario: 直接测试 ignore band gate target
- **WHEN** 测试代码传入 `delta` 和 `ignore_delta_eps`
- **THEN** helper MUST 返回二值 target 和 target valid mask
- **AND** `abs(delta)` 不大于阈值的位置 MUST 被标记为无效

#### Scenario: 直接测试 context marginal mask
- **WHEN** 测试代码请求为目标模态构造 `context_marginal` mask
- **THEN** helper MUST 返回不含目标模态的上下文 mask 和加入目标模态后的 mask
- **AND** 两个 mask MUST 遵守可用模态约束和最小保留模态数量

### Requirement: Teacher-prior CRAF 组件注册
项目 MUST 通过现有组件注册和默认组件导入边界暴露 teacher-prior CRAF 所需组件。新增模型、gate、loss、KD loss 和 helper MUST 可由配置或窄模块导入复用，并且不得要求复制训练脚本。

#### Scenario: 注册 PriorResidualGate 或 gate factory
- **WHEN** 配置选择 `gate_type: prior_residual_sigmoid`
- **THEN** 系统 MUST 能构建 prior residual gate
- **AND** 构建失败时错误信息 MUST 包含 gate 类型和可用 gate 类型

#### Scenario: 注册 teacher-prior CRAF 模型入口
- **WHEN** 配置选择 teacher-prior CRAF 所需模型类型
- **THEN** `MODELS` 注册表 MUST 能构建对应模型
- **AND** `import_default_components()` 后可用模型列表 MUST 包含该模型或继续包含可承载该 gate 的 `craf_fusion`

#### Scenario: 注册 prior 和 KD loss
- **WHEN** 配置显式启用 prior regularization 或 reliability-weighted KD
- **THEN** 系统 MUST 能通过现有 loss/distillation 构建边界调用对应 loss
- **AND** 关闭这些 loss 时训练流程 MUST 不构建无用组件

### Requirement: Teacher loader 组件边界
teacher encoder loader MUST 以窄模块函数或可测试组件提供。loader MUST 不依赖训练循环内部局部变量，并 MUST 能在单元测试中用合成 checkpoint 验证 key mapping、strict 模式和冻结策略。

#### Scenario: 单元测试直接调用 teacher loader
- **WHEN** 测试用合成 teacher checkpoint 调用 teacher loader
- **THEN** loader MUST 返回每模态 load summary
- **AND** loader MUST 能在没有 dataloader 或 trainer 的情况下运行

#### Scenario: strict 模式抛出清晰错误
- **WHEN** strict loader 遇到 shape mismatch
- **THEN** loader MUST 抛出包含模态、checkpoint 路径和 mismatch key 的错误

### Requirement: 默认导入保持轻量
新增 teacher-prior CRAF 组件 MUST 遵守现有轻量导入约束。导入 `kd_sensing.registries` MUST 不急切导入训练器、dataset 或 checkpoint 文件；默认组件导入 MUST 仍由构建流程显式触发。

#### Scenario: 轻量导入 registry 不触发训练模块
- **WHEN** 开发者执行 `import kd_sensing.registries`
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入 teacher registry 构建脚本或 trainer 模块

#### Scenario: 构建 CRAF 前导入默认组件
- **WHEN** 构建流程调用 `import_default_components()` 后再查询 `MODELS`
- **THEN** teacher-prior CRAF 相关内置模型或 gate 所在模块 MUST 已完成注册

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

