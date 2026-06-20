## ADDED Requirements

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

## REMOVED Requirements

### Requirement: RadarStudent 模型结构
**Reason**: Radar lightweight baseline 不再作为 standalone whole-model 注册名维护。
**Migration**: 使用 `configs/radar/lightweight.yaml`，其主模型为 `modular_sequence`。

### Requirement: RadarStudent 当前训练兼容
**Reason**: 该要求允许 `radar_lightweight` 或等价注册名作为 current primary model；迁移后 current primary model 必须是 `modular_sequence`。
**Migration**: 在 migrated config tests 中验证 radar-only supervised/adaptation runtime 仍兼容。

### Requirement: RadarStudent 默认 GRU 层数
**Reason**: 默认 GRU 层数不再属于 `radar_lightweight` whole-model 契约。
**Migration**: 在 `configs/radar/lightweight.yaml` 的 `representation_core.num_layers` 中维护。
