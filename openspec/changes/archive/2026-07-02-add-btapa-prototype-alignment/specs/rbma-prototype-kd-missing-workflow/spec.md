## ADDED Requirements

### Requirement: Prototype target type switch
现有 beam prototype alignment MUST 保留旧逻辑，并 MUST 通过配置选择 onehot、旧 soft target 或 BTAPA beam-soft target。默认旧 V3 配置 MUST 不被新 BTAPA 配置覆盖。

#### Scenario: 旧 V3 prototype 配置保持不变
- **WHEN** 用户加载 `configs/scene31/main_v3_strong_reliability_proto.yaml`
- **THEN** 配置 MUST 继续使用 strong encoder、weighted_sum reliability fusion、missing modality mask 和旧 beam prototype alignment
- **AND** 配置 MUST 不启用 RBMA、JEPA、KD 或 full auxiliary loss

#### Scenario: BTAPA 配置启用新 target
- **WHEN** 用户加载 BTAPA Scene31 配置
- **THEN** 配置 MUST 设置 `use_beam_topology_proto=true` 和 `proto_target_type=beam_soft`
- **AND** 输出 run name MUST 与旧 V3 baseline 区分
