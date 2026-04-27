## ADDED Requirements

### Requirement: RadarTeacher 蒸馏角色配置
系统 MUST 支持在 radar-only KD 配置中将 `radar_teacher` 同时配置为 frozen teacher 和可训练 student。该配置 MUST 保持 RadarTeacher 既有输入输出契约，MUST 不要求图像输入，并且 MUST 不改变 radar no-KD 基线中 `radar_teacher` 作为主模型训练的语义。

#### Scenario: 构建 radar logits KD 模型
- **WHEN** 配置指定 `model.teacher.type: radar_teacher`、`model.student.type: radar_teacher` 且 `distillation.type: logits_kd`
- **THEN** 系统 MUST 通过模型注册表构建两个 RadarTeacher 实例
- **AND** teacher MUST 在训练中被冻结并加载配置指定的 RadarTeacher checkpoint
- **AND** student MUST 作为可训练主模型参与 optimizer 更新

#### Scenario: 构建 radar RKD 模型
- **WHEN** 配置指定 `model.teacher.type: radar_teacher`、`model.student.type: radar_teacher` 且 `distillation.type: rkd`
- **THEN** 系统 MUST 通过模型注册表构建两个 RadarTeacher 实例
- **AND** teacher 和 student 的 forward 输出 MUST 提供 RKD 所需的输出特征
- **AND** RKD MUST 使用 RadarTeacher 输出特征计算样本间距离和角度关系损失

#### Scenario: 保持 radar no-KD 基线语义
- **WHEN** 配置指定 `distillation.type: no_kd` 且 `distillation.teacher_model_name: null`
- **THEN** 系统 MUST 不构建或加载 teacher checkpoint
- **AND** 系统 MUST 继续直接训练 `radar_teacher` 主模型作为 radar-only 基线
