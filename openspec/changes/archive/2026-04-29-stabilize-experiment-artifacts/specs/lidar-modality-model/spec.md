## MODIFIED Requirements

### Requirement: LiDAR KD 兼容性
LiDAR-only teacher/student MUST 与现有 logits KD 和 RKD distiller 兼容。默认 LiDAR KD 配置 MUST 使用 `lidar_teacher` 作为 frozen teacher，并使用 `lidar_student` 作为可训练 student。默认 LiDAR teacher 和 student 配置 MUST 都使用 `gru_params: [64, 64, 1]`。

#### Scenario: LiDAR logits KD
- **WHEN** 用户运行 LiDAR-only logits KD 配置
- **THEN** 系统 MUST 构建 frozen `lidar_teacher` 和可训练 `lidar_student`
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

#### Scenario: LiDAR RKD
- **WHEN** 用户运行 LiDAR-only RKD 配置
- **THEN** 系统 MUST 构建 frozen `lidar_teacher` 和可训练 `lidar_student`
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练
- **AND** teacher/student output feature 维度 MUST 在默认配置中保持一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

## ADDED Requirements

### Requirement: LiDAR 单模态默认 GRU 层数
默认 LiDAR teacher 和 LiDAR student 单模态配置 MUST 使用一层 GRU，以便与当前 LiDAR 配置、README 和测试保持一致。

#### Scenario: lidar_teacher 默认 GRU 层数
- **WHEN** 用户通过默认 LiDAR teacher no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

#### Scenario: lidar_student 默认 GRU 层数
- **WHEN** 用户通过默认 LiDAR student no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1
