## MODIFIED Requirements

### Requirement: GPS student 模型
系统 MUST 提供已注册的 `gps_student` 模型，用于 lightweight GPS-only beam prediction。该模型 MUST 接收 `[B, T, 3]` GPS-Rel-Polar 特征序列，使用比 teacher 更轻量的 `GpsFeatureExtractor` 或投影层、LayerNorm、GRU temporal modeling 和小型 classifier 输出 beam logits。默认 GPS student 配置 MUST 使用 `gru_params: [64, 64, 2]`。

#### Scenario: 构建 gps_student
- **WHEN** 配置中指定 `model.student.type: gps_student`
- **THEN** 模型注册表 MUST 能构建 `gps_student`
- **AND** 构建参数 MUST 支持 `gps_input_size`、`feature_size`、`num_classes`、`gru_params` 和可选宽度控制参数
- **AND** GPS-Rel-Polar 配置中的 `gps_input_size` MUST 为 3

#### Scenario: gps_student forward contract
- **WHEN** `gps_student` 接收形状为 `[B, T, 3]` 的 GPS 输入张量
- **THEN** 模型 MUST 返回 `(pred, input_features, output_features)`
- **AND** `pred` 的形状 MUST 为 `[B, T, num_classes]`
- **AND** `input_features` 的形状 MUST 为 `[B, T, feature_size]`
- **AND** `output_features` 的 batch 和 sequence 维度 MUST 与输入一致

#### Scenario: gps_student 默认 GRU 层数
- **WHEN** 用户通过默认 GPS student 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 2]`
- **AND** 模型的 `GRU.num_layers` MUST 为 2

### Requirement: GPS KD 兼容性
GPS-only teacher/student MUST 与现有 logits KD 和 RKD distiller 兼容。默认 GPS KD 配置 MUST 使用 `gps_teacher` 作为 frozen teacher，并使用 `gps_student` 作为可训练 student。默认 GPS teacher 和 student 配置 MUST 都使用 `gru_params: [64, 64, 2]`。

#### Scenario: GPS logits KD
- **WHEN** 用户运行 GPS-only logits KD 配置
- **THEN** 系统 MUST 构建 frozen `gps_teacher` 和可训练 `gps_student`
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: GPS RKD
- **WHEN** 用户运行 GPS-only RKD 配置
- **THEN** 系统 MUST 构建 frozen `gps_teacher` 和可训练 `gps_student`
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练
- **AND** teacher/student output feature 维度 MUST 在默认配置中保持一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`
