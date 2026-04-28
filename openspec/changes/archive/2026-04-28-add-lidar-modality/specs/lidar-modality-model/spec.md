## ADDED Requirements

### Requirement: LiDARFeatureExtractor 结构
系统 MUST 提供 `LidarFeatureExtractor`，用于从 LiDAR BEV 序列中提取每个时隙的固定长度特征。该 feature extractor MUST 接收形状为 `(batch, sequence, channels, height, width)` 的 LiDAR BEV 张量，并输出 `(batch, sequence, feature_size)`。

#### Scenario: LiDAR feature extractor 前向输出
- **WHEN** `LidarFeatureExtractor` 接收形状为 `(B, T, C, H, W)` 的 LiDAR BEV 输入
- **THEN** 输出 MUST 为形状 `(B, T, feature_size)` 的特征张量
- **AND** 输出 feature 维 MUST 等于构造参数 `n_feature` 或 `feature_size`

#### Scenario: LiDAR feature extractor 注册与导出
- **WHEN** 开发者从 `kd_sensing.models.lidar` 或 `kd_sensing.models` 导入 LiDAR feature extractor
- **THEN** 系统 MUST 暴露 `LidarFeatureExtractor`
- **AND** 系统 MUST 通过模型注册表提供 `lidar_feature_extractor` 或等价配置构建入口

### Requirement: LiDARTeacher 模型结构
系统 MUST 提供已注册的 `lidar_teacher` 模型，用于 LiDAR-only beam prediction。该模型的公开实现类和包导出名称 MUST 为 `LidarModalityNet`，并 MUST 接收 LiDAR BEV 序列张量，使用 `LidarFeatureExtractor`、LayerNorm、GRU temporal modeling、attention 或等价时序增强模块和 MLP classifier 输出 beam logits。

#### Scenario: 按配置构建 LiDARTeacher
- **WHEN** 配置中指定 `model.teacher.type: lidar_teacher` 或 `model.student.type: lidar_teacher`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `LidarModalityNet` 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`lidar_channels` 和 attention 相关参数

#### Scenario: LiDARTeacher 前向输出契约
- **WHEN** `LidarModalityNet` 接收形状为 `(batch, sequence, channels, height, width)` 的 LiDAR BEV 输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: LiDARTeacher 参数校验
- **WHEN** `gru_params` 不包含 `[input_size, hidden_size, num_layers]` 三个值，或 `gru_input_size` 不等于 `feature_size`
- **THEN** `LidarModalityNet` MUST 在构建时抛出明确异常

### Requirement: LiDARStudent 模型结构
系统 MUST 提供已注册的 `lidar_student` 模型，用于 LiDAR-only lightweight beam prediction。该模型的公开实现类和包导出名称 MUST 为 `LidarStudentModalityNet`，并 MUST 使用轻量 CNN embedding、adaptive pooling、特征投影、LayerNorm、GRU temporal modeling 和 MLP classifier 输出 beam logits。

#### Scenario: 按配置构建 LiDARStudent
- **WHEN** 配置中指定 `model.student.type: lidar_student`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `LidarStudentModalityNet` 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`lidar_channels` 和 `width_multiplier`

#### Scenario: LiDARStudent 前向输出契约
- **WHEN** `LidarStudentModalityNet` 接收形状为 `(batch, sequence, channels, height, width)` 的 LiDAR BEV 输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: LiDARStudent 不依赖固定 BEV 尺寸
- **WHEN** `LidarStudentModalityNet` 对每个 LiDAR 时隙提取空间特征
- **THEN** 系统 MUST 使用 adaptive pooling 将空间特征聚合为固定长度向量
- **AND** 模型 MUST 不依赖固定 flatten 输入尺寸

### Requirement: LiDAR-only 输入准备
系统 MUST 提供 LiDAR-only 输入准备路径，从 batch 中读取 `lidar`，按现有预测窗口规则补齐未来占位帧，并将结果传给 LiDAR 模型。

#### Scenario: 准备 LiDAR-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: lidar`
- **THEN** 系统 MUST 使用 batch 中的 `lidar` 构造 LiDAR 输入
- **AND** 系统 MUST 不要求图像、雷达或 GPS 输入参与模型 forward

#### Scenario: LiDAR 预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** LiDAR-only 输入 MUST 包含最近 8 个 LiDAR 时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 继续使用最后 `num_pred + 1` 个输出时隙与标签对齐

### Requirement: LiDAR-only 基线配置
项目 MUST 提供 LiDAR-only 配置，用于训练和评估 LiDAR teacher baseline 和 lightweight student baseline。配置 MUST 使用 `experiment.task: lidar`，并通过 `lidar_teacher` 或 `lidar_student` 构建主模型。

#### Scenario: 启动 LiDAR teacher no-KD 训练
- **WHEN** 用户使用 LiDAR teacher no-KD 配置运行训练入口
- **THEN** 系统 MUST 构建 `lidar_teacher` 作为被优化的主模型
- **AND** 训练流程 MUST 完成 forward、task loss、backward、optimizer step、validation 和 checkpoint 保存

#### Scenario: 启动 LiDAR student no-KD 训练
- **WHEN** 用户使用 LiDAR student no-KD 配置运行训练入口
- **THEN** 系统 MUST 构建 `lidar_student` 作为被优化的主模型
- **AND** 系统 MUST 不构建或加载 frozen teacher
- **AND** 系统 MUST 只使用 LiDAR 输入完成 forward

### Requirement: LiDAR KD 兼容性
LiDAR-only teacher/student MUST 与现有 logits KD 和 RKD distiller 兼容。默认 LiDAR KD 配置 MUST 使用 `lidar_teacher` 作为 frozen teacher，并使用 `lidar_student` 作为可训练 student。默认 LiDAR teacher 和 student 配置 MUST 都使用 `gru_params: [64, 64, 2]`。

#### Scenario: LiDAR logits KD
- **WHEN** 用户运行 LiDAR-only logits KD 配置
- **THEN** 系统 MUST 构建 frozen `lidar_teacher` 和可训练 `lidar_student`
- **AND** 系统 MUST 使用任务 loss 与 logits KL 蒸馏 loss 的加权结果进行训练
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: LiDAR RKD
- **WHEN** 用户运行 LiDAR-only RKD 配置
- **THEN** 系统 MUST 构建 frozen `lidar_teacher` 和可训练 `lidar_student`
- **AND** 系统 MUST 使用任务 loss 与关系蒸馏 loss 的加权结果进行训练
- **AND** teacher/student output feature 维度 MUST 在默认配置中保持一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`
