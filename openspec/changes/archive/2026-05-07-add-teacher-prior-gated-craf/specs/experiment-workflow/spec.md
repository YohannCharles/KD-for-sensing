## ADDED Requirements

### Requirement: Teacher-prior CRAF stage workflow
训练流程 MUST 支持 teacher-prior CRAF 的 Stage 1、Stage 2 和 Stage 3 工作流，并 MUST 继续复用统一训练入口、输出目录、checkpoint、TensorBoard 和 `train_log.json` 语义。

#### Scenario: Stage 1 训练单模态 teacher
- **WHEN** 用户运行任一单模态 teacher-prior Stage 1 配置
- **THEN** 系统 MUST 使用对应单模态数据和 teacher 模型训练
- **AND** 输出目录 MUST 保存 best checkpoint、last checkpoint、最终配置和可供 teacher registry 读取的验证指标

#### Scenario: Stage 2 初始化发生在 optimizer 前
- **WHEN** 用户运行 Stage 2 teacher-init prior 配置
- **THEN** 系统 MUST 在构建 optimizer 前加载 teacher encoder 并应用冻结策略
- **AND** optimizer MUST 只包含 `requires_grad=True` 的参数

#### Scenario: Stage 3 checkpoint 加载后应用 finetune 策略
- **WHEN** 用户运行 Stage 3 selective fine-tuning 配置
- **THEN** 系统 MUST 先加载 Stage 2 checkpoint
- **AND** 系统 MUST 再应用选择性冻结/解冻策略并构建参数组 optimizer

### Requirement: Teacher registry build command
项目 MUST 提供可命令行运行的 teacher registry 构建流程。该流程 MUST 使用 conda 环境中的 Python 运行，并 MUST 能从配置或命令行参数指定 teacher root、输出路径、prior 模式和场景。

#### Scenario: 从 teacher 根目录生成 registry
- **WHEN** 用户运行 teacher registry 构建命令并指定 teacher root
- **THEN** 系统 MUST 扫描或读取五个单模态 teacher 输出目录
- **AND** 系统 MUST 写出 teacher registry JSON 到指定路径

#### Scenario: registry 写出路径父目录不存在
- **WHEN** 用户指定的 teacher registry 输出路径父目录不存在
- **THEN** 系统 MUST 创建父目录
- **AND** 系统 MUST 不覆盖 unrelated 输出文件

### Requirement: Teacher-prior CRAF optimizer 参数组
训练流程 MUST 支持 Stage 3 参数组 optimizer。参数组 MUST 按 fusion/head/gate/strong encoder/weak encoder 或等价角色划分，并 MUST 在训练日志中记录每组学习率和参数量。

#### Scenario: Stage 3 参数组非空
- **WHEN** Stage 3 配置解冻 GPS 和 mmWave encoder
- **THEN** strong encoder 参数组 MUST 包含 GPS 和 mmWave encoder 参数
- **AND** weak encoder 参数组 MUST 不包含 frozen image、radar 或 LiDAR 参数
- **AND** fusion、head 和 gate 参数组 MUST 非空

#### Scenario: 冻结参数不进入 optimizer
- **WHEN** 某个 encoder 参数 `requires_grad=False`
- **THEN** optimizer 参数组 MUST 不包含该参数
- **AND** 训练日志 MUST 记录该 encoder 为 frozen

### Requirement: Teacher-prior CRAF validation subsets
验证流程 MUST 支持对 CRAF 模型运行显式模态组合评估。该能力 MUST 只在模型支持 force modality mask 且配置启用时运行。

#### Scenario: 运行 strong-only 和 weak-only 验证
- **WHEN** 配置启用 `evaluation.modality_subsets`
- **THEN** 验证流程 MUST 使用 force modality mask 分别评估 strong-only 和 weak-only 组合
- **AND** strong-only MUST 对应 GPS+mmWave
- **AND** weak-only MUST 对应 image+radar+LiDAR

#### Scenario: 非 CRAF 模型跳过模态组合验证
- **WHEN** 模型不支持 `supports_force_modality_mask`
- **THEN** 验证流程 MUST 跳过模态组合评估
- **AND** 默认验证指标 MUST 仍正常产出

### Requirement: Teacher-prior CRAF smoke tests
项目 MUST 提供面向 teacher-prior CRAF 的短训练和定向测试路径。测试命令 MUST 使用 `conda run -n kd_mm_beam` 环境约束。

#### Scenario: PriorResidualGate 初始化测试
- **WHEN** 开发者运行 CRAF 定向测试
- **THEN** 测试 MUST 覆盖 prior residual gate 初始化后 gate 接近 prior
- **AND** 测试 MUST 覆盖 unavailable modality mask

#### Scenario: Stage 2/3 workflow smoke test
- **WHEN** 开发者运行 Stage 2 或 Stage 3 synthetic smoke test
- **THEN** 训练流程 MUST 完成 forward、loss、backward、optimizer step、validation 和 checkpoint 保存
- **AND** 测试 MUST 验证冻结或选择性解冻策略生效
