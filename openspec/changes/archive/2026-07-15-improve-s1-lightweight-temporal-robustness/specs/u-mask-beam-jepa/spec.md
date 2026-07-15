## ADDED Requirements

### Requirement: U-Mask S1-compatible temporal pooling
U-MaskBeamJEPA MUST 支持新的 `temporal_pooling` opt-in 字典，在 `fusion_type=supervised_router` 下消费现有 temporal metadata，生成 per-modality pooled features 并复用 current router/head。该新配置 MUST 不接受或映射旧 S1-S4 `temporal_router_type` 名称。

#### Scenario: Temporal metadata 进入模型
- **WHEN** U-Mask 配置启用 temporal pooling 且 batch 包含 `[B,T,M]` modality temporal mask
- **THEN** forward MUST 使用与 model modality order 一致的 mask 进行聚合
- **AND** output diagnostics MUST 保留 modality temporal mask、temporal mask、modality availability 和 pooling metadata

#### Scenario: 旧 S1-S4 名称继续拒绝
- **WHEN** current config 请求 `s1_temporalagg_modality`、`s2_pertime_modality`、`s3_two_level` 或 `s4_global`
- **THEN** model/config validation MUST 继续拒绝该值
- **AND** 系统 MUST 不静默映射到新的 temporal pooling 行为

### Requirement: Temporal pooling 与 router metadata 可审计
启用 temporal pooling、mask statistics 或 coverage shrinkage 时，`training_strategy_metadata()` MUST 记录配置类型、有效参数量、消费的 mask statistics、teacher/ranking 开关和 shrinkage 上限。模型输出 MUST 继续兼容 current `ModelOutput` adapter 与共享训练/评估 runtime。

#### Scenario: Startup summary 记录轻量增量
- **WHEN** S1-compatible temporal pooling model 被构建
- **THEN** startup/model metadata MUST 记录 total/trainable params 和 temporal pooling params
- **AND** metadata MUST 能区分 masked mean、fixed recency、gap-aware residual 和 shrinkage enabled 状态

#### Scenario: Shared runtime 无专用分支
- **WHEN** 训练或评估 S1-compatible config
- **THEN** batch MUST 通过共享 task input/runtime 进入 U-Mask
- **AND** trainer、validator 和 evaluator MUST 不新增 S1 专用 forward loop
