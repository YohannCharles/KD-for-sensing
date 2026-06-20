# radar-student-model Specification

## Purpose
定义 radar lightweight 模型结构、注册名和当前训练兼容行为，确保雷达轻量分支可用于 supervised/adaptation 训练。
## Requirements
### Requirement: RadarStudent 轻量特征提取
RadarStudent MUST 使用轻量雷达特征提取路径，避免复用 RadarTeacher 的固定 flatten 全连接 embedding 作为主干。轻量特征提取 MUST 使用 depthwise separable convolution block 或等价轻量卷积结构，并通过 adaptive pooling 生成固定长度帧特征。

#### Scenario: 不依赖固定 flatten 尺寸
- **WHEN** RadarStudent 对每个雷达时隙提取空间特征
- **THEN** 系统 MUST 使用 adaptive pooling 将空间特征聚合为固定长度向量
- **AND** 模型 MUST 不依赖 `64 * 8 * 4` 这类 teacher flatten 输入尺寸

#### Scenario: 输出 feature size 对齐
- **WHEN** `feature_size` 配置为 64
- **THEN** RadarStudent 的投影层 MUST 为每个时隙输出 64 维输入特征

### Requirement: Radar lightweight canonical 配置使用 modular_sequence
Radar lightweight canonical 配置 MUST 使用 `modular_sequence`、`radar_cnn` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 Radar lightweight whole-model 注册名。

#### Scenario: 构建 radar lightweight 配置
- **WHEN** 用户加载 `configs/radar/lightweight.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.radar.type` MUST 为 `radar_cnn`
- **AND** lightweight 差异 MUST 通过配置参数表达

### Requirement: Radar lightweight legacy names are removed
Radar student/lightweight legacy whole-model 注册名 MUST 被 removed guard 拒绝，并指向 modular radar baseline。

#### Scenario: 请求 radar lightweight legacy 注册名
- **WHEN** 用户请求 `radar_student` 或 `radar_lightweight`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + radar_cnn + single_gru`

