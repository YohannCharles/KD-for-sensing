## ADDED Requirements

### Requirement: RadarStudent no-KD 实验配置
项目 MUST 提供 radar-only lightweight student no-KD 配置，用于直接训练 `radar_student` 并评估轻量雷达模型在无蒸馏条件下的表现。该配置 MUST 使用 `experiment.task: radar`，MUST 不加载 teacher checkpoint，并 MUST 复用统一训练、验证、评估和输出目录语义。

#### Scenario: 使用 no-KD 启动 RadarStudent 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/student_no_kd.yaml`
- **THEN** 系统 MUST 构建 `radar_student` 作为可训练主模型
- **AND** 系统 MUST 不构建或加载 frozen teacher
- **AND** 系统 MUST 只使用雷达输入完成 forward

## MODIFIED Requirements

### Requirement: 配置驱动实验
项目 MUST 提供配置文件驱动的训练、评估和预处理入口。配置 MUST 覆盖数据路径、CSV 文件名、模态类型、teacher/student 模型、KD 模式、训练超参数、优化器、调度器、输出目录和随机种子。

#### Scenario: 使用配置启动 image-only 训练
- **WHEN** 用户通过新 CLI 传入 image-only 训练配置
- **THEN** 系统 MUST 构建 image-only dataset、teacher/student 模型、KD/loss、optimizer 和 scheduler，并进入训练流程

#### Scenario: 使用配置启动 image+radar 训练
- **WHEN** 用户通过新 CLI 传入 fusion 训练配置
- **THEN** 系统 MUST 构建同时包含图像和雷达输入的 dataset、fusion teacher/student 模型、KD/loss、optimizer 和 scheduler，并进入训练流程

#### Scenario: 使用配置启动 radar-only 训练
- **WHEN** 用户通过新 CLI 传入 radar-only 训练配置
- **THEN** 系统 MUST 构建包含雷达输入的 dataset、配置指定的 radar-only 主模型、loss、optimizer 和 scheduler，并进入训练流程
- **AND** 训练流程 MUST 支持 `radar_teacher` baseline 和 `radar_student` lightweight student
- **AND** 训练流程 MUST 不要求模型接收图像输入

#### Scenario: 使用配置启动 radar-only 评估
- **WHEN** 用户通过新 CLI 传入 radar-only 评估配置和 radar-only 模型权重
- **THEN** 系统 MUST 构建配置指定的 radar-only 模型并只使用雷达输入完成评估
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

### Requirement: Radar-only KD 实验配置
项目 MUST 提供 radar-only KD 配置，使 radar-only 实验能够通过配置选择 `logits_kd` 和 `rkd` 蒸馏模式。KD 配置 MUST 使用 `experiment.task: radar`，MUST 通过 `radar_teacher` 构建 frozen teacher，MUST 通过 `radar_student` 构建可训练 student，MUST 配置可解析的 RadarTeacher checkpoint 来源，并且 MUST 继续复用统一训练入口、loss、optimizer、scheduler、验证指标和输出目录语义。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/logits_kd.yaml`
- **THEN** 系统 MUST 构建 `logits_kd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_student` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练

#### Scenario: 使用 RKD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/rkd.yaml`
- **THEN** 系统 MUST 构建 `rkd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_student` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练

#### Scenario: 使用默认 RadarTeacher checkpoint
- **WHEN** 用户未覆盖 radar KD 配置中的 teacher 权重字段
- **THEN** 系统 MUST 从 radar teacher no-KD 训练输出目录解析 teacher checkpoint
- **AND** 该默认路径 MUST 对应 `outputs/radar_no_kd/checkpoints/best.pth`

#### Scenario: 覆盖 RadarTeacher checkpoint
- **WHEN** 用户通过命令行覆盖 `paths.weights_dir` 或 `distillation.teacher_model_name`
- **THEN** 系统 MUST 使用覆盖后的值解析 radar teacher checkpoint
- **AND** 系统 MUST 保持其它 radar-only KD 配置语义不变
