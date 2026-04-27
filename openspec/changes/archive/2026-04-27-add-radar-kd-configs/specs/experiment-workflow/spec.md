## ADDED Requirements

### Requirement: Radar-only KD 实验配置
项目 MUST 提供 radar-only KD 配置，使 radar-only 实验能够通过配置选择 `no_kd`、`logits_kd` 和 `rkd` 三种模式。KD 配置 MUST 使用 `experiment.task: radar`，MUST 通过 `radar_teacher` 构建 teacher 和 student，MUST 配置可解析的 RadarTeacher checkpoint 来源，并且 MUST 继续复用统一训练入口、loss、optimizer、scheduler、验证指标和输出目录语义。

#### Scenario: 使用 logits KD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/logits_kd.yaml`
- **THEN** 系统 MUST 构建 `logits_kd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_teacher` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练

#### Scenario: 使用 RKD 启动 radar-only 训练
- **WHEN** 用户通过训练入口传入 `configs/radar/rkd.yaml`
- **THEN** 系统 MUST 构建 `rkd` 蒸馏组件
- **AND** 系统 MUST 构建 frozen `radar_teacher` teacher 和可训练 `radar_teacher` student
- **AND** 系统 MUST 只使用雷达输入完成 teacher/student forward
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练

#### Scenario: 使用默认 RadarTeacher checkpoint
- **WHEN** 用户未覆盖 radar KD 配置中的 teacher 权重字段
- **THEN** 系统 MUST 从 radar no-KD 训练输出目录解析 teacher checkpoint
- **AND** 该默认路径 MUST 对应 `outputs/radar_no_kd/checkpoints/best.pth`

#### Scenario: 覆盖 RadarTeacher checkpoint
- **WHEN** 用户通过命令行覆盖 `paths.weights_dir` 或 `distillation.teacher_model_name`
- **THEN** 系统 MUST 使用覆盖后的值解析 radar KD teacher checkpoint
- **AND** 系统 MUST 保持其它 radar-only KD 配置语义不变
