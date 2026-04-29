## MODIFIED Requirements

### Requirement: RadarStudent 蒸馏兼容
`RadarStudentModalityNet` MUST 与现有 radar-only 训练、验证、评估和蒸馏流程兼容。系统 MUST 能将 `RadarModalityNet` 作为 frozen teacher，将 `RadarStudentModalityNet` 作为可训练 student，并复用 logits KD 与 RKD distiller。默认 radar teacher 和 radar student 单模态 KD 配置 MUST 使用 `gru_params: [64, 64, 1]`。

#### Scenario: 使用 logits KD 训练 RadarStudent
- **WHEN** radar-only KD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** 训练流程 MUST 只使用雷达输入完成 teacher 和 student forward
- **AND** logits KD MUST 使用 teacher/student logits 计算蒸馏损失
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

#### Scenario: 使用 RKD 训练 RadarStudent
- **WHEN** radar-only RKD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** `RadarStudentModalityNet` MUST 返回可用于 RKD 的 output_features
- **AND** 默认配置 MUST 保持 teacher/student output hidden size 一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

## ADDED Requirements

### Requirement: RadarStudent 默认 GRU 层数
默认 RadarStudent 单模态配置 MUST 使用一层 GRU，以便与当前 radar-only lightweight student 配置、README 和测试保持一致。

#### Scenario: radar_student no-KD 默认 GRU 层数
- **WHEN** 用户通过默认 radar student no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1
