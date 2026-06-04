## MODIFIED Requirements

### Requirement: RadarStudent 模型结构
系统 MUST 提供已注册的 radar lightweight 模型，用于 radar-only lightweight beam prediction。该模型 MUST 接收 RA/DA 拼接后的雷达序列张量，使用轻量 CNN embedding、adaptive pooling、特征投影、LayerNorm、GRU temporal modeling 和 MLP classifier 输出 beam logits。该模型不再作为 KD student 定义。

#### Scenario: 按配置构建 radar lightweight
- **WHEN** 配置中指定 radar lightweight primary model
- **THEN** 系统 MUST 通过模型注册表构建对应轻量雷达模型实例
- **AND** 构建参数 MUST 支持 `feature_size`、`num_classes`、`gru_params` 和 `radar_channels`

#### Scenario: radar lightweight 前向输出契约
- **WHEN** radar lightweight 模型接收形状为 `(batch, sequence, channels, height, width)` 的雷达输入张量
- **THEN** 模型 MUST 返回可用于 beam supervised loss 和 metrics 的 logits
- **AND** 系统 MUST 不要求输出 RKD 专用特征

## REMOVED Requirements

### Requirement: RadarStudent 蒸馏兼容
**Reason**: RadarStudent 不再作为 KD student。
**Migration**: 使用 radar lightweight supervised 配置。

#### Scenario: 使用 logits KD 训练 RadarStudent
- **WHEN** 用户运行旧 radar KD 配置
- **THEN** 系统 MUST 拒绝该配置
- **AND** 系统 MUST 不构建 distiller

