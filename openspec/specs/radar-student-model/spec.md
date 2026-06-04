# radar-student-model Specification

## Purpose
定义 radar lightweight 模型结构、注册名和当前训练兼容行为，确保雷达轻量分支可用于 supervised/adaptation 训练。
## Requirements
### Requirement: RadarStudent 模型结构
系统 MUST 提供已注册的 `radar_student` 模型，用于 radar-only lightweight beam prediction。该模型的公开实现类和包导出名称 MUST 为 `RadarStudentModalityNet`，并 MUST 接收 RA/DA 拼接后的雷达序列张量，使用轻量 CNN embedding、adaptive pooling、特征投影、LayerNorm、GRU temporal modeling 和 MLP classifier 输出 beam logits。该模型不再作为 KD student 定义。

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
- **AND** 系统 MUST 不要求输出 RKD 专用特征

#### Scenario: RadarStudent 参数校验
- **WHEN** `gru_params` 不包含 `[input_size, hidden_size, num_layers]` 三个值，或 `gru_input_size` 不等于 `feature_size`
- **THEN** `RadarStudentModalityNet` MUST 在构建时抛出明确异常

#### Scenario: Radar student 公共导出命名
- **WHEN** 开发者从 `kd_sensing.models.radar` 或 `kd_sensing.models` 导入 radar student 类
- **THEN** 系统 MUST 暴露 `RadarStudentModalityNet`
- **AND** 仓库内代码、测试和主文档 MUST 不再引用旧 radar student 类名

### Requirement: RadarStudent 轻量特征提取
RadarStudent MUST 使用轻量雷达特征提取路径，避免复用 RadarTeacher 的固定 flatten 全连接 embedding 作为主干。轻量特征提取 MUST 使用 depthwise separable convolution block 或等价轻量卷积结构，并通过 adaptive pooling 生成固定长度帧特征。

#### Scenario: 不依赖固定 flatten 尺寸
- **WHEN** RadarStudent 对每个雷达时隙提取空间特征
- **THEN** 系统 MUST 使用 adaptive pooling 将空间特征聚合为固定长度向量
- **AND** 模型 MUST 不依赖 `64 * 8 * 4` 这类 teacher flatten 输入尺寸

#### Scenario: 输出 feature size 对齐
- **WHEN** `feature_size` 配置为 64
- **THEN** RadarStudent 的投影层 MUST 为每个时隙输出 64 维输入特征

### Requirement: RadarStudent 当前训练兼容
`RadarStudentModalityNet` MUST 与 radar-only supervised/adaptation 训练、验证和评估流程兼容。旧 radar logits KD 或 RKD 配置 MUST 在配置解析阶段失败。

#### Scenario: 使用 supervised 训练 RadarStudent
- **WHEN** radar-only lightweight 配置指定 `model.primary.type: radar_lightweight` 或等价注册名
- **THEN** 训练流程 MUST 只使用雷达输入完成 primary model forward
- **AND** 训练流程 MUST 使用 supervised/adaptation loss 更新该主模型

#### Scenario: 旧 radar KD 配置被拒绝
- **WHEN** 用户运行旧 radar-only logits KD 或 RKD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 distiller

### Requirement: RadarStudent 默认 GRU 层数
默认 RadarStudent 单模态配置 MUST 使用一层 GRU，以便与当前 radar-only lightweight student 配置、README 和测试保持一致。

#### Scenario: radar lightweight 默认 GRU 层数
- **WHEN** 用户通过默认 radar lightweight 配置构建模型
- **THEN** 配置中的 `gru_params` MUST 为 `[64, 64, 1]`
- **AND** 模型的 `GRU.num_layers` MUST 为 1
