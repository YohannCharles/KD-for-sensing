## MODIFIED Requirements

### Requirement: RadarStudent 模型结构
系统 MUST 提供已注册的 `radar_student` 模型，用于 radar-only lightweight beam prediction。该模型的公开实现类和包导出名称 MUST 为 `RadarStudentModalityNet`，并 MUST 接收 RA/DA 拼接后的雷达序列张量，使用轻量 CNN embedding、adaptive pooling、特征投影、LayerNorm、GRU temporal modeling 和 MLP classifier 输出 beam logits。

#### Scenario: 按配置构建 RadarStudent
- **WHEN** 配置中指定 `model.student.type: radar_student`
- **THEN** 系统 MUST 通过 `MODELS` 注册表构建 `RadarStudentModalityNet` 实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params`、`radar_channels`

#### Scenario: RadarStudent 前向输出契约
- **WHEN** `RadarStudentModalityNet` 接收形状为 `(batch, sequence, channels, height, width)` 的雷达输入张量
- **THEN** 模型 MUST 返回 `(pred, features, output_features)`
- **AND** `pred` 的形状 MUST 为 `(batch, sequence, num_classes)`
- **AND** `features` 的形状 MUST 为 `(batch, sequence, feature_size)`
- **AND** `output_features` 的 batch 与 sequence 维度 MUST 与输入一致

#### Scenario: RadarStudent 参数校验
- **WHEN** `gru_params` 不包含 `[input_size, hidden_size, num_layers]` 三个值，或 `gru_input_size` 不等于 `feature_size`
- **THEN** `RadarStudentModalityNet` MUST 在构建时抛出明确异常

#### Scenario: Radar student 公共导出命名
- **WHEN** 开发者从 `kd_sensing.models.radar` 或 `kd_sensing.models` 导入 radar student 类
- **THEN** 系统 MUST 暴露 `RadarStudentModalityNet`
- **AND** 仓库内代码、测试和主文档 MUST 不再引用旧 radar student 类名

### Requirement: RadarStudent 蒸馏兼容
`RadarStudentModalityNet` MUST 与现有 radar-only 训练、验证、评估和蒸馏流程兼容。系统 MUST 能将 `RadarModalityNet` 作为 frozen teacher，将 `RadarStudentModalityNet` 作为可训练 student，并复用 logits KD 与 RKD distiller。默认 radar student 配置 MUST 使用 `gru_params: [64, 64, 2]`。

#### Scenario: 使用 logits KD 训练 RadarStudent
- **WHEN** radar-only KD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** 训练流程 MUST 只使用雷达输入完成 teacher 和 student forward
- **AND** logits KD MUST 使用 teacher/student logits 计算蒸馏损失
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`

#### Scenario: 使用 RKD 训练 RadarStudent
- **WHEN** radar-only RKD 配置指定 `model.teacher.type: radar_teacher` 且 `model.student.type: radar_student`
- **THEN** `RadarStudentModalityNet` MUST 返回可用于 RKD 的 output_features
- **AND** 默认配置 MUST 保持 teacher/student output hidden size 一致
- **AND** teacher 和 student 配置的 `gru_params` MUST 为 `[64, 64, 2]`
