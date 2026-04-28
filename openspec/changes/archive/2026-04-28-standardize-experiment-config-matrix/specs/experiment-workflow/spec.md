## ADDED Requirements

### Requirement: 单模态 canonical 配置矩阵
项目 MUST 为每个受支持单模态 `image`、`radar`、`gps` 和 `lidar` 提供统一命名的 canonical 配置矩阵。每个单模态目录 MUST 包含 `teacher_no_kd.yaml`、`student_no_kd.yaml`、`logits_kd.yaml` 和 `rkd.yaml`。canonical 配置 MUST 使用统一训练、验证、评估、loss、optimizer、scheduler、checkpoint 和输出目录语义。

#### Scenario: 单模态 teacher no-KD 配置
- **WHEN** 开发者加载 `configs/<modality>/teacher_no_kd.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 设置 `distillation.teacher_model_name: null`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_teacher`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_teacher_no_kd`

#### Scenario: 单模态 student no-KD 配置
- **WHEN** 开发者加载 `configs/<modality>/student_no_kd.yaml`
- **THEN** 配置 MUST 使用该模态对应的 `experiment.task`
- **AND** 配置 MUST 设置 `distillation.type: no_kd`
- **AND** 配置 MUST 设置 `distillation.teacher_model_name: null`
- **AND** 配置 MUST 将被训练主模型配置为对应 `<modality>_student`
- **AND** 配置的 `experiment.name` 和 `output.run_name` MUST 使用 `<modality>_student_no_kd`

#### Scenario: 单模态 logits KD 配置
- **WHEN** 开发者加载 `configs/<modality>/logits_kd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: logits_kd`
- **AND** 配置 MUST 构建 frozen `<modality>_teacher`
- **AND** 配置 MUST 构建可训练 `<modality>_student`
- **AND** 配置 MUST 默认解析对应 canonical teacher no-KD 输出中的 `best.pth`

#### Scenario: 单模态 RKD 配置
- **WHEN** 开发者加载 `configs/<modality>/rkd.yaml`
- **THEN** 配置 MUST 设置 `distillation.type: rkd`
- **AND** 配置 MUST 构建 frozen `<modality>_teacher`
- **AND** 配置 MUST 构建可训练 `<modality>_student`
- **AND** 配置 MUST 提供 `rkd_pairs_per_anchor`、`rkd_distance_weight` 和 `rkd_angle_weight`
- **AND** 配置 MUST 默认解析对应 canonical teacher no-KD 输出中的 `best.pth`

### Requirement: 单模态 legacy no-KD 入口兼容
项目 MUST 保留现有 `configs/<modality>/no_kd.yaml` 入口作为兼容配置，并 MUST 在文档中说明其历史语义和推荐替代入口。legacy 入口不得改变 canonical 配置矩阵的语义。

#### Scenario: image legacy no-KD 保持 student baseline
- **WHEN** 用户运行 `configs/image/no_kd.yaml`
- **THEN** 系统 MUST 继续训练 `image_student`
- **AND** 文档 MUST 引导新实验优先使用 `configs/image/student_no_kd.yaml`

#### Scenario: radar GPS LiDAR legacy no-KD 保持 teacher baseline
- **WHEN** 用户运行 `configs/radar/no_kd.yaml`、`configs/gps/no_kd.yaml` 或 `configs/lidar/no_kd.yaml`
- **THEN** 系统 MUST 继续训练对应 teacher baseline
- **AND** 文档 MUST 引导新实验优先使用对应 `teacher_no_kd.yaml`

### Requirement: teacher/student 角色不得受原脚本残留影响
配置驱动流程 MUST 以 YAML 中的 `model.student` 作为 no-KD 时的被训练主模型，并 MUST 只在 `distillation.type` 非 `no_kd` 时构建 frozen teacher。默认 canonical student baseline 和 KD 配置 MUST 使用 lightweight student，不得默认使用 teacher-as-student 残留。

#### Scenario: no-KD 只训练配置中的主模型
- **WHEN** 配置设置 `distillation.type: no_kd`
- **THEN** 训练流程 MUST 不构建或加载 frozen teacher
- **AND** optimizer MUST 只更新 `model.student` 构建出的主模型

#### Scenario: canonical student baseline 使用 lightweight student
- **WHEN** 开发者加载任意 canonical `student_no_kd.yaml`
- **THEN** `model.student.type` MUST 为对应 lightweight student 注册名
- **AND** `model.student.type` MUST NOT 等于对应 teacher 注册名

#### Scenario: canonical KD 使用 teacher 蒸馏 student
- **WHEN** 开发者加载任意 canonical `logits_kd.yaml` 或 `rkd.yaml`
- **THEN** `model.teacher.type` MUST 为对应 teacher 注册名
- **AND** `model.student.type` MUST 为对应 lightweight student 注册名
- **AND** teacher 和 student 的输出 hidden size MUST 对齐以支持 RKD

### Requirement: canonical 配置命名与输出目录一致
canonical 配置 MUST 使用可预测的实验名、run name 和默认 teacher checkpoint 来源。默认路径 MUST 便于用户按 teacher baseline -> student baseline/KD 的顺序运行实验，并 MUST 支持命令行覆盖。

#### Scenario: canonical run name 与文件语义一致
- **WHEN** 开发者加载任意 canonical 配置
- **THEN** `experiment.name` MUST 与不含 `.yaml` 的文件 stem 一致
- **AND** `output.run_name` MUST 与 `experiment.name` 一致

#### Scenario: canonical KD 默认读取 teacher baseline 输出
- **WHEN** 用户未覆盖 canonical KD 配置中的 teacher 权重字段
- **THEN** 系统 MUST 从对应 canonical `teacher_no_kd` 输出目录解析 teacher checkpoint
- **AND** 默认 checkpoint 文件名 MUST 为 `best.pth`

#### Scenario: canonical KD checkpoint 可覆盖
- **WHEN** 用户通过命令行覆盖 `paths.weights_dir` 或 `distillation.teacher_model_name`
- **THEN** 系统 MUST 使用覆盖后的 teacher checkpoint 来源
- **AND** 系统 MUST 保持该配置的 teacher/student 模型角色不变
