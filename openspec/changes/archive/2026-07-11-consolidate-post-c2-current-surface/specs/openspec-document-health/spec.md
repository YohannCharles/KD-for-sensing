## MODIFIED Requirements

### Requirement: current JEPA 合法语境不被旧路线 guard 误判
项目健康护栏 MUST 允许 current JEPA pretraining、MMW mean-context reuse、checkpoint extraction 和禁用 condition-id 的安全诊断语境，同时 MUST 将 GPS-query/predictive/visual-shortcut 作为 retired wording。历史说明或 removed delta 中出现旧名称时不得误报为回流。

#### Scenario: Current mean JEPA wording 被允许
- **WHEN** current spec 描述 `gps_conditioned_jepa` pretraining、`jepa_context_image` 或 `pooling: mean`
- **THEN** retired-route guard MUST 不误报
- **AND** current wording MUST 不要求 GPS-query/predictive pooler

#### Scenario: GPS-query active wording 被拒绝
- **WHEN** current docs/specs 将 GPS-query、predictive JEPA 或 shortcut benchmark 描述为 current config、baseline 或推荐入口
- **THEN** wording guard MUST 失败
- **AND** historical/removed migration context MAY 保留

#### Scenario: forbidden condition 字段诊断被允许
- **WHEN** current source 或 spec 记录 `condition_id_consumed=false`、`blocked_condition_fields` 或 `forbidden_condition_fields`
- **THEN** 健康护栏 MUST 将其解释为安全边界
- **AND** 只有把 condition id 描述为模型输入/current router 时才应失败
