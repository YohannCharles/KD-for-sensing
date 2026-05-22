# gps-modality-model Specification

## Purpose
定义 GPS teacher/student 模型、特征模式和配置兼容行为，确保 GPS 分支能在单模态与 fusion 训练中稳定复用。
## Requirements
### Requirement: GPS teacher 模型
系统 MUST 提供已注册的 `gps_teacher` 模型，用于 GPS-only beam prediction。该模型 MUST 接收 `[B, T, 3]` GPS-Rel-Polar 特征序列，使用 `GpsFeatureExtractor` 提取每个时隙 embedding，经过 LayerNorm、GRU temporal modeling、时序增强模块和 MLP classifier 后输出 beam logits。

#### Scenario: 构建 gps_teacher
- **WHEN** 配置中指定 `model.teacher.type: gps_teacher` 或 `model.student.type: gps_teacher`
- **THEN** 模型注册表 MUST 能构建 `gps_teacher`
- **AND** 构建参数 MUST 支持 `gps_input_size`、`feature_size`、`num_classes` 和 `gru_params`
- **AND** GPS-Rel-Polar 配置中的 `gps_input_size` MUST 为 3

#### Scenario: gps_teacher forward contract
- **WHEN** `gps_teacher` 接收形状为 `[B, T, 3]` 的 GPS 输入张量
- **THEN** 模型 MUST 返回 `(pred, input_features, output_features)`
- **AND** `pred` 的形状 MUST 为 `[B, T, num_classes]`
- **AND** `input_features` 的形状 MUST 为 `[B, T, feature_size]`
- **AND** `output_features` 的 batch 和 sequence 维度 MUST 与输入一致

### Requirement: GPS student 模型
系统 MUST 提供已注册的 `gps_student` 模型，用于 lightweight GPS-only beam prediction。该模型 MUST 接收 `[B, T, 3]` GPS-Rel-Polar 特征序列，使用比 teacher 更轻量的 `GpsFeatureExtractor` 或投影层、LayerNorm、GRU temporal modeling 和小型 classifier 输出 beam logits。默认 GPS student 配置 MUST 使用 `gru_params: [64, 64, 1]`。

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
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

### Requirement: GPS 模型参数校验
GPS teacher 和 GPS student MUST 校验 `gru_params` 和输入维度配置。`gru_params` MUST 包含 `[input_size, hidden_size, num_layers]`，且 `input_size` MUST 等于 `feature_size`。

#### Scenario: 非法 gru_params 长度
- **WHEN** 用户用长度不是 3 的 `gru_params` 构建 `gps_teacher` 或 `gps_student`
- **THEN** 系统 MUST 抛出配置错误

#### Scenario: GRU 输入维度不匹配
- **WHEN** 用户配置的 `gru_params[0]` 不等于 `feature_size`
- **THEN** 系统 MUST 抛出包含实际维度和期望维度的错误

### Requirement: GPS-only 任务输入
训练、验证和评估流程 MUST 支持 `experiment.task: gps`。GPS-only 任务 MUST 只准备 GPS 输入和 label，不要求 image 或 radar 输入。

#### Scenario: GPS-only 训练 forward
- **WHEN** 用户通过训练入口运行 `experiment.task: gps` 的配置
- **THEN** 系统 MUST 从 batch 中读取 `gps`
- **AND** 系统 MUST 按预测窗口规则补齐未来 GPS 占位时隙
- **AND** GPS 输入特征维度 MUST 为 3
- **AND** 系统 MUST 调用 GPS 模型完成 forward

#### Scenario: GPS-only 评估 forward
- **WHEN** 用户通过评估入口运行 `experiment.task: gps` 的配置和 GPS 模型权重
- **THEN** 系统 MUST 构建配置指定的 GPS 模型并只使用 GPS 输入完成评估
- **AND** GPS 输入特征维度 MUST 为 3
- **AND** 系统 MUST 保存 Top-K、DBA 和 loss 指标

### Requirement: GPS KD 兼容性
GPS-only teacher/student MUST 与现有 logits KD 和 RKD distiller 兼容。默认 GPS KD 配置 MUST 使用 `gps_teacher` 作为 frozen teacher，并使用 `gps_student` 作为可训练 student。默认 GPS teacher 和 student 配置 MUST 都使用 `gru_params: [64, 64, 1]`。

#### Scenario: GPS logits KD
- **WHEN** 用户运行 GPS-only logits KD 配置
- **THEN** 系统 MUST 构建 frozen `gps_teacher` 和可训练 `gps_student`
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

#### Scenario: GPS RKD
- **WHEN** 用户运行 GPS-only RKD 配置
- **THEN** 系统 MUST 构建 frozen `gps_teacher` 和可训练 `gps_student`
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练
- **AND** teacher/student output feature 维度 MUST 在默认配置中保持一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

### Requirement: GPS teacher 默认 GRU 层数
默认 GPS teacher 单模态配置 MUST 使用一层 GRU，以便与当前单模态配置、README 和测试保持一致。

#### Scenario: gps_teacher 默认 GRU 层数
- **WHEN** 用户通过默认 GPS teacher no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1
