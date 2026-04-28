## MODIFIED Requirements

### Requirement: RadarTeacher 模型结构
系统 MUST 提供已注册的 `radar_teacher` 模型，用于 radar-only beam prediction。该模型的公开实现类和包导出名称 MUST 为 `RadarModalityNet`，并 MUST 接收 RA/DA 拼接后的雷达序列张量，使用任务特定 CNN embedding 提取每个时隙的低维特征，经过 LayerNorm、GRU temporal modeling、MHA residual prediction module 和 MLP classifier 后输出 beam logits。

#### Scenario: 按配置构建 RadarTeacher
- **WHEN** 配置中指定 `model.student.type: radar_teacher` 或 `model.teacher.type: radar_teacher`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `RadarModalityNet` 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`radar_channels` 和 `num_heads`

#### Scenario: RadarTeacher 前向输出契约
- **WHEN** `RadarModalityNet` 接收形状为 `(batch, sequence, channels, height, width)` 的雷达输入张量
- **THEN** 模型 MUST 返回 `(pred, features, enhanced_seq_out)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `enhanced_seq_out` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: MHA 参数校验
- **WHEN** `gru_hidden_size` 不能被 `num_heads` 整除
- **THEN** `RadarModalityNet` MUST 在构建时抛出明确异常，说明 MHA head 配置无效

#### Scenario: Radar teacher 公共导出命名
- **WHEN** 开发者从 `kd_sensing.models.radar` 或 `kd_sensing.models` 导入 radar teacher 类
- **THEN** 系统 MUST 暴露 `RadarModalityNet`
- **AND** 仓库内代码、测试和主文档 MUST 不再引用旧 radar teacher 类名

### Requirement: RadarTeacher 蒸馏角色配置
系统 MUST 支持在 radar-only KD 配置中将 `radar_teacher` 作为 frozen teacher，并将轻量 `radar_student` 作为默认可训练 student。该配置 MUST 保持 `RadarModalityNet` 既有输入输出契约，MUST 不要求图像输入，并且 MUST 不改变 radar no-KD baseline 中 `radar_teacher` 作为主模型训练的语义。系统 MAY 继续允许用户通过显式配置将 `radar_teacher` 同时作为 teacher 和 student，用于兼容 teacher-as-student 实验。

#### Scenario: 构建 radar logits KD 模型
- **WHEN** 配置指定 `model.teacher.type: radar_teacher`、`model.student.type: radar_student` 且 `distillation.type: logits_kd`
- **THEN** 系统 MUST 通过模型注册表构建 frozen `RadarModalityNet` teacher 和可训练 `RadarStudentModalityNet` student
- **AND** teacher MUST 在训练中被冻结并加载配置指定的 RadarTeacher checkpoint
- **AND** student MUST 作为可训练主模型参与 optimizer 更新

#### Scenario: 构建 radar RKD 模型
- **WHEN** 配置指定 `model.teacher.type: radar_teacher`、`model.student.type: radar_student` 且 `distillation.type: rkd`
- **THEN** 系统 MUST 通过模型注册表构建 frozen `RadarModalityNet` teacher 和可训练 `RadarStudentModalityNet` student
- **AND** teacher 和 student 的 forward 输出 MUST 提供 RKD 所需的输出特征
- **AND** RKD MUST 使用 teacher/student 输出特征计算样本间距离和角度关系损失

#### Scenario: 保持 radar no-KD 基线语义
- **WHEN** 配置指定 `distillation.type: no_kd`、`distillation.teacher_model_name: null` 且 `model.student.type: radar_teacher`
- **THEN** 系统 MUST 不构建或加载 teacher checkpoint
- **AND** 系统 MUST 继续直接训练 `radar_teacher` 主模型作为 radar-only teacher baseline

#### Scenario: 兼容显式 teacher-as-student 配置
- **WHEN** 用户显式配置 `model.teacher.type: radar_teacher`、`model.student.type: radar_teacher` 且启用 radar-only KD
- **THEN** 系统 MUST 继续支持构建两个 `RadarModalityNet` 实例
- **AND** frozen teacher 和可训练 student MUST 维持既有输入输出契约
