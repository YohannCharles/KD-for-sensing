## ADDED Requirements

### Requirement: GPS canonical 配置使用 modular_sequence
GPS strong、lightweight、supervised 和当前保留的 GPS ablation canonical 配置 MUST 使用 `modular_sequence`、`gps_mlp` encoder、projector、`single_gru` representation core 和 `beam_head`，而不是旧 GPS whole-model 注册名。

#### Scenario: 构建 GPS strong/supervised 配置
- **WHEN** 用户加载 `configs/gps/strong.yaml` 或 `configs/gps/supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** `model.primary.encoders.gps.type` MUST 为 `gps_mlp`
- **AND** 模型 forward MUST 只要求 GPS batch 输入和 beam labels

#### Scenario: 构建 GPS lightweight 配置
- **WHEN** 用户加载 `configs/gps/lightweight.yaml`
- **THEN** 系统 MUST 构建 `modular_sequence` GPS-only 模型
- **AND** lightweight 差异 MUST 通过配置参数表达，而不是通过 `gps_lightweight` whole-model 注册名表达

### Requirement: GPS legacy model names are removed
GPS legacy whole-model 注册名 MUST 被 removed guard 拒绝。该规则覆盖 `gps_teacher`、`gps_student`、`gps_strong`、`gps_lightweight`、`gps_sequence_baseline` 的退役场景；若某名称仍需作为 current baseline，必须在 design 中单独说明并保留 focused tests。

#### Scenario: 请求 GPS legacy 注册名
- **WHEN** 用户请求构建退役 GPS 注册名
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence + gps_mlp + single_gru`

## REMOVED Requirements

### Requirement: GPS teacher 模型
**Reason**: GPS teacher whole-model 路线已被 `modular_sequence + gps_mlp` canonical GPS baseline 替代；`gps_teacher` 旧名已经属于 removed guard 语义。
**Migration**: 使用 `configs/gps/strong.yaml` 或 `configs/gps/supervised.yaml`，其 `model.primary.type` 为 `modular_sequence`。

### Requirement: GPS student 模型
**Reason**: GPS student/lightweight whole-model 路线已被 `modular_sequence + gps_mlp` lightweight config 替代。
**Migration**: 使用 `configs/gps/lightweight.yaml`，并通过 encoder/core/head 参数表达 lightweight 差异。

### Requirement: GPS 模型参数校验
**Reason**: 该要求绑定 `gps_teacher` / `gps_student` 构造参数；迁移后参数校验由 `gps_mlp` encoder、`single_gru` core 和 `modular_sequence` 组件构建负责。
**Migration**: 在 modular GPS config tests 中校验 `gps_input_size=3`、encoder output dim、core input dim 和 beam head class 数。

### Requirement: GPS teacher 默认 GRU 层数
**Reason**: 默认 GRU 层数不再属于 `gps_teacher` whole-model 契约。
**Migration**: 在 `configs/gps/{strong,supervised,lightweight}.yaml` 的 `representation_core.num_layers` 中维护。
