## ADDED Requirements

### Requirement: RadarStudent 模型结构
系统 MUST 提供已注册的 `radar_student` 模型，用于 radar-only lightweight beam prediction。该模型 MUST 接收 RA/DA 拼接后的雷达序列张量，使用轻量 CNN embedding、adaptive pooling、特征投影、LayerNorm、GRU temporal modeling 和 MLP classifier 输出 beam logits。

#### Scenario: 按配置构建 RadarStudent
- **WHEN** 配置中指定 `model.student.type: radar_student`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 RadarStudent 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`radar_channels`

#### Scenario: RadarStudent 前向输出契约
- **WHEN** RadarStudent 接收形状为 `(batch, sequence, channels, height, width)` 的雷达输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: RadarStudent 参数校验
- **WHEN** `gru_params` 不包含 `[input_size, hidden_size, num_layers]` 三个值，或 `gru_input_size` 不等于 `feature_size`
- **THEN** RadarStudent MUST 在构建时抛出明确异常

### Requirement: RadarStudent 轻量特征提取
RadarStudent MUST 使用轻量雷达特征提取路径，避免复用 RadarTeacher 的固定 flatten 全连接 embedding 作为主干。轻量特征提取 MUST 使用 depthwise separable convolution block 或等价轻量卷积结构，并通过 adaptive pooling 生成固定长度帧特征。

#### Scenario: 不依赖固定 flatten 尺寸
- **WHEN** RadarStudent 对每个雷达时隙提取空间特征
- **THEN** 系统 MUST 使用 adaptive pooling 将空间特征聚合为固定长度向量
- **AND** 模型 MUST 不依赖 `64 * 8 * 4` 这类 teacher flatten 输入尺寸

#### Scenario: 输出 feature size 对齐
- **WHEN** `feature_size` 配置为 64
- **THEN** RadarStudent 的投影层 MUST 为每个时隙输出 64 维输入特征

### Requirement: RadarStudent 蒸馏兼容
RadarStudent MUST 与现有 radar-only 训练、验证、评估和蒸馏流程兼容。系统 MUST 能将 RadarTeacher 作为 frozen teacher，将 RadarStudent 作为可训练 student，并复用 logits KD 与 RKD distiller。

#### Scenario: 使用 logits KD 训练 RadarStudent
- **WHEN** radar-only KD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** 训练流程 MUST 只使用雷达输入完成 teacher 和 student forward
- **AND** logits KD MUST 使用 teacher/student logits 计算蒸馏损失

#### Scenario: 使用 RKD 训练 RadarStudent
- **WHEN** radar-only RKD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** RadarStudent MUST 返回可用于 RKD 的 output_features
- **AND** 默认配置 MUST 保持 teacher/student output hidden size 一致
