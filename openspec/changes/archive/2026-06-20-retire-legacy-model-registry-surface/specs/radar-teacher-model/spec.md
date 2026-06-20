## ADDED Requirements

### Requirement: Radar strong canonical 配置使用 modular_sequence
Radar strong 和 supervised canonical 配置 MUST 使用 `modular_sequence`、`radar_cnn` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 Radar whole-model 注册名。

#### Scenario: 构建 radar strong/supervised 配置
- **WHEN** 用户加载 `configs/radar/strong.yaml` 或 `configs/radar/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.radar.type` MUST 为 `radar_cnn`
- **AND** radar-only task runtime MUST 继续从 batch 中准备 radar 输入并适配 beam logits

### Requirement: Radar strong legacy names are removed
Radar teacher/strong legacy whole-model 注册名 MUST 被 removed guard 拒绝，并指向 modular radar baseline。

#### Scenario: 请求 radar strong legacy 注册名
- **WHEN** 用户请求 `radar_teacher` 或 `radar_strong`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + radar_cnn + single_gru`

## REMOVED Requirements

### Requirement: RadarTeacher 模型结构
**Reason**: Radar strong baseline 不再作为 standalone whole-model 注册名维护。
**Migration**: 使用 `modular_sequence + radar_cnn + single_gru + beam_head`。

### Requirement: Radar-only 基线配置
**Reason**: 该要求指定 radar-only 配置必须构建 `radar_teacher`，与本 change 的 modular canonical 入口冲突。
**Migration**: 保留 `configs/radar/strong.yaml` 和 `configs/radar/supervised.yaml` 路径，但其主模型改为 `modular_sequence`。
