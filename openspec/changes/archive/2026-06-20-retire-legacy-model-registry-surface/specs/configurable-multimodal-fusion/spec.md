## ADDED Requirements

### Requirement: Legacy fusion whole-model routes are retired
普通 fusion baseline MUST 优先使用 `modular_sequence` 组件化路径。旧 `fusion_lightweight` 和无 current config 依赖的 `fusion_strong` whole-model 注册名 MUST 被 removed guard 拒绝；保留的 fusion whole-model 注册名必须有 current spec 或 whole-model exception 理由。

#### Scenario: radar+GPS supervised fusion 使用 modular_sequence
- **WHEN** 用户加载 `configs/fusion/radar_gps_supervised.yaml`
- **THEN** 最终配置的 `model.primary.type` MUST 为 `modular_sequence`
- **AND** 配置 MUST 使用 `radar_cnn`、`gps_mlp`、projectors、`early_concat_gru` 或等价 current representation core
- **AND** fusion task runtime MUST 继续只准备启用模态的 batch 输入

#### Scenario: 请求 legacy fusion 注册名
- **WHEN** 用户请求 `fusion_lightweight` 或 `fusion_strong`
- **THEN** registry MUST 抛出 removed component 错误
- **AND** 错误信息 MUST 建议使用 `modular_sequence` fusion 配置

#### Scenario: current fusion whole-model exceptions 不受影响
- **WHEN** 用户配置 current 保留的 `cls_token_transformer_fusion` 或 `token_transformer_fusion`
- **THEN** 系统 MUST 继续按对应 current spec 或 config 构建模型
- **AND** 本 change MUST 不改变这些保留模型的 forward/output 契约
