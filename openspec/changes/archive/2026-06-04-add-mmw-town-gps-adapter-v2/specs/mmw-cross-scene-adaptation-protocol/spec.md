## ADDED Requirements

### Requirement: MMW Town GPS-only v2 protocols
系统 MUST 为 MMW Town GPS-only v2 支持 `source_other_three`、`target_adapt_beambench` 和 `within_scene_train` 三类协议。协议输出 MUST 明确标记 scene、source scenes、target scene、support/query/test split 和 strict eligibility。

#### Scenario: source_other_three 留一场景评估
- **WHEN** runner 执行 `source_other_three`
- **THEN** 系统 MUST 每次选择一个 MMW Town scene 作为 target
- **AND** 系统 MUST 使用其它三个 scene 训练 source model
- **AND** target scene label MUST 只用于最终评估

#### Scenario: target_adapt_beambench 使用 support/query
- **WHEN** runner 执行 `target_adapt_beambench`
- **THEN** 系统 MUST 使用其它三个 scene 训练 source backbone
- **AND** target support set MUST 只来自 target adaptation pool
- **AND** target query/test MUST 不参与 adapter 初始化、优化或模型选择

#### Scenario: within_scene_train 标记为上界
- **WHEN** runner 执行 `within_scene_train`
- **THEN** 系统 MUST 将该 protocol 标记为同场景上界或 sanity protocol
- **AND** summary MUST NOT 将其作为跨场景泛化结论

### Requirement: MMW Town GPS v2 support selection
系统 MUST 为 v2 target adaptation 支持 `temporal_first`、`random` 和 `trajectory` support selection。默认 support mode MUST 为 `temporal_first`，support 数量可由 `support_ratio` 或 `support_num` 控制。

#### Scenario: temporal_first 默认支持集
- **WHEN** 用户未显式指定 support mode
- **THEN** 系统 MUST 使用 `temporal_first`
- **AND** support manifest MUST 记录 sample id、timestamp/order key、target label、scene、selection mode 和 seed

#### Scenario: support_num 覆盖 support_ratio
- **WHEN** 用户同时配置 `support_num` 和 `support_ratio`
- **THEN** 系统 MUST 使用 `support_num` 或抛出清晰的优先级错误
- **AND** metadata MUST 记录实际使用的 support count
