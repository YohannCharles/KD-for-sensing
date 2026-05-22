# mmwave-modality-model Specification

## Purpose
定义 mmWave power vector 模型、feature extractor、scaler 和输入语义。
## Requirements
### Requirement: MmWaveFeatureExtractor 结构
系统 MUST 提供 `MmWaveFeatureExtractor`，用于从 mmWave 64 维 receive-power 特征序列中提取每个时隙的固定长度 embedding。该 feature extractor MUST 接收形状为 `(batch, sequence, 64)` 的 mmWave 张量，并输出 `(batch, sequence, feature_size)`。

#### Scenario: mmWave feature extractor 前向输出
- **WHEN** `MmWaveFeatureExtractor` 接收形状为 `(B, T, 64)` 的 mmWave 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: mmWave feature extractor 注册与导出
- **WHEN** 开发者从 `kd_sensing.models.mmwave` 或 `kd_sensing.models` 导入 mmWave feature extractor
- **THEN** 系统 MUST 暴露 `MmWaveFeatureExtractor`
- **AND** 系统 MUST 通过模型注册表提供 `mmwave_feature_extractor` 或等价配置构建入口

### Requirement: mmWave teacher 模型结构
系统 MUST 提供已注册的 `mmwave_teacher` 模型，用于 mmWave-only beam prediction。该模型的公开实现类和包导出名称 MUST 为 `MmWaveModalityNet`，并 MUST 接收 mmWave 64 维特征序列张量，使用 `MmWaveFeatureExtractor`、LayerNorm、GRU temporal modeling、attention 或等价时序增强模块和 MLP classifier 输出 beam logits。

#### Scenario: 按配置构建 mmWave teacher
- **WHEN** 配置中指定 `model.teacher.type: mmwave_teacher` 或 `model.student.type: mmwave_teacher`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `MmWaveModalityNet` 实例
- **AND** 构建参数 MUST 支持 `mmwave_input_size`、`feature_size`、`num_classes`、`gru_params` 和 attention 相关参数
- **AND** 默认 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave teacher 前向输出契约
- **WHEN** `MmWaveModalityNet` 接收形状为 `(batch, sequence, 64)` 的 mmWave 输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

### Requirement: mmWave student 模型结构
系统 MUST 提供已注册的 `mmwave_student` 模型，用于 mmWave-only lightweight beam prediction。该模型的公开实现类和包导出名称 MUST 为 `MmWaveStudentModalityNet`，并 MUST 使用轻量 MLP embedding、LayerNorm、GRU temporal modeling 和小型 classifier 输出 beam logits。

#### Scenario: 按配置构建 mmWave student
- **WHEN** 配置中指定 `model.student.type: mmwave_student`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `MmWaveStudentModalityNet` 实例
- **AND** 构建参数 MUST 支持 `mmwave_input_size`、`feature_size`、`num_classes`、`gru_params` 和可选宽度控制参数
- **AND** 默认 `mmwave_input_size` MUST 为 64

#### Scenario: mmWave student 前向输出契约
- **WHEN** `MmWaveStudentModalityNet` 接收形状为 `(batch, sequence, 64)` 的 mmWave 输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

### Requirement: mmWave 模型参数校验
mmWave teacher 和 mmWave student MUST 校验 `gru_params` 和输入维度配置。`gru_params` MUST 包含 `[input_size, hidden_size, num_layers]`，且 `input_size` MUST 等于 `feature_size`；`mmwave_input_size` MUST 等于 dataset 输出的 64 维特征。

#### Scenario: 非法 gru_params 长度
- **WHEN** 用户用长度不是 3 的 `gru_params` 构建 `mmwave_teacher` 或 `mmwave_student`
- **THEN** 系统 MUST 抛出配置错误

#### Scenario: GRU 输入维度不匹配
- **WHEN** 用户配置的 `gru_params[0]` 不等于 `feature_size`
- **THEN** 系统 MUST 抛出包含实际维度和期望维度的错误

#### Scenario: mmWave 输入维度不匹配
- **WHEN** mmWave 模型收到最后一维不是 64 或不等于 `mmwave_input_size` 的输入张量
- **THEN** 系统 MUST 抛出包含实际输入维度和期望维度的错误

### Requirement: mmWave-only 输入准备
系统 MUST 提供 mmWave-only 输入准备路径，从 batch 中读取 `mmwave`，按现有预测窗口规则补齐未来占位时隙，并将结果传给 mmWave 模型。

#### Scenario: 准备 mmWave-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: mmwave`
- **THEN** 系统 MUST 使用 batch 中的 `mmwave` 构造 mmWave 输入
- **AND** 系统 MUST 不要求图像、雷达、GPS 或 LiDAR 输入参与模型 forward

#### Scenario: mmWave 预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** mmWave-only 输入 MUST 包含最近 8 个 mmWave 历史时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam

### Requirement: mmWave-only 基线配置
项目 MUST 提供 mmWave-only 配置，用于训练和评估 mmWave teacher baseline 和 lightweight student baseline。配置 MUST 使用 `experiment.task: mmwave`，并通过 `mmwave_teacher` 或 `mmwave_student` 构建主模型。

#### Scenario: 启动 mmWave teacher no-KD 训练
- **WHEN** 用户使用 mmWave teacher no-KD 配置运行训练入口
- **THEN** 系统 MUST 构建 `mmwave_teacher` 作为被优化的主模型
- **AND** 训练流程 MUST 完成 forward、task loss、backward、optimizer step、validation 和 checkpoint 保存

#### Scenario: 启动 mmWave student no-KD 训练
- **WHEN** 用户使用 mmWave student no-KD 配置运行训练入口
- **THEN** 系统 MUST 构建 `mmwave_student` 作为被优化的主模型
- **AND** 系统 MUST 不构建或加载 frozen teacher
- **AND** 系统 MUST 只使用 mmWave 输入完成 forward

### Requirement: mmWave KD 兼容性
mmWave-only teacher/student MUST 与现有 logits KD 和 RKD distiller 兼容。默认 mmWave KD 配置 MUST 使用 `mmwave_teacher` 作为 frozen teacher，并使用 `mmwave_student` 作为可训练 student。默认 mmWave teacher 和 student 配置 MUST 都使用 `gru_params: [64, 64, 1]`。

#### Scenario: mmWave logits KD
- **WHEN** 用户运行 mmWave-only logits KD 配置
- **THEN** 系统 MUST 构建 frozen `mmwave_teacher` 和可训练 `mmwave_student`
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

#### Scenario: mmWave RKD
- **WHEN** 用户运行 mmWave-only RKD 配置
- **THEN** 系统 MUST 构建 frozen `mmwave_teacher` 和可训练 `mmwave_student`
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练
- **AND** teacher/student output feature 维度 MUST 在默认配置中保持一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 1]`

### Requirement: mmWave 单模态默认 GRU 层数
默认 mmWave teacher 和 mmWave student 单模态配置 MUST 使用一层 GRU，以便与当前 image、radar、GPS 和 LiDAR 单模态配置保持一致。

#### Scenario: mmwave_teacher 默认 GRU 层数
- **WHEN** 用户通过默认 mmWave teacher no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1

#### Scenario: mmwave_student 默认 GRU 层数
- **WHEN** 用户通过默认 mmWave student no-KD 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1
