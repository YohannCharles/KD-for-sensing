## ADDED Requirements

### Requirement: MMW Sionna path 数据可用性与字段映射
系统 MUST 在 MMW 数据准备、manifest 或巡检阶段记录 Sionna channel/path 文件中的 path-level physical propagation parameters 可用性。字段识别 MUST 支持配置化 `data.field_map`，不得依赖单一硬编码字段名。

#### Scenario: 记录 path-level 字段摘要
- **WHEN** frame manifest 或 inspect 工具匹配到 V2I channel/path 文件
- **THEN** 系统 MUST 记录是否存在 path gain、delay、AoD/AoA azimuth/zenith、valid path mask 和 optional Tx/Rx/CAV/RSU pose
- **AND** 系统 MUST 记录字段 shape、dtype 或等价摘要
- **AND** 系统 MUST 不把原始 path tensor 复制进源码控制的 manifest

#### Scenario: 使用 data.field_map 覆盖字段名
- **WHEN** Sionna path 文件字段名与默认候选不一致
- **THEN** 用户 MUST 能通过 `data.field_map` 指定 gain、delay、AoD/AoA、mask 和 pose 字段映射
- **AND** 系统 MUST 在 metadata 中记录实际使用的字段映射

#### Scenario: path 数据缺失时保留样本可诊断
- **WHEN** 某个样本缺少 path-level parameters 但仍有合法 sensing inputs 和 beam label
- **THEN** 系统 MUST 允许该样本继续用于不依赖 path supervision 的训练或评估
- **AND** manifest 或 dataset metadata MUST 记录 path unavailable reason

### Requirement: MMW path-level split 与评估边界
MMW scenario/town/weather split MUST 将 path-level labels 和 descriptors 视为 auxiliary target 或 diagnostic data，而不是 sensing input。target adaptation 与 target_test 的防泄漏边界 MUST 对 path fields 生效。

#### Scenario: source split 可构造 path labels
- **WHEN** source train split 有可用 path parameters
- **THEN** 系统 MAY 基于 source path descriptors fit path semantic label artifact
- **AND** artifact MUST 记录 source town/scenario/weather、fit sample count 和 unavailable count

#### Scenario: target test path labels 只用于离线评价
- **WHEN** target test 样本可构造 path_semantic_label 或 path_descriptor
- **THEN** evaluation MAY 使用这些字段计算 path diagnostics
- **AND** target adaptation MUST NOT 使用 target test path fields 选择 threshold、更新 prototype 或计算训练 loss

#### Scenario: leave-one-town/scenario/weather 报告 path 分布
- **WHEN** 系统生成 MMW LOSO summary
- **THEN** summary MUST 能报告 source-target path class histogram 或 unavailable reason
- **AND** summary MUST 将 leave-one-town-out、leave-one-scenario-out 和 weather-shift protocol 的 path diagnostics 与 run metadata 关联
