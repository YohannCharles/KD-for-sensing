## ADDED Requirements

### Requirement: MMW profile 与结构候选 provenance 必须匹配
MMW training、checkpoint metadata、evaluation worker 和 summary MUST 记录 training profile id、profile fingerprint、T2 design candidate id 与 resolved recipe fingerprint。比较同一方法的多 seed 或同一 summary 行时，任一这些身份不一致 MUST fail closed。

#### Scenario: 拒绝混合 H0/H4 或不同候选
- **WHEN** summary 接收 profile 或 candidate fingerprint 不一致的输入
- **THEN** summary MUST 标记该比较不可用或抛出校验错误
- **AND** 不得通过缺失字段、默认值或人工补值继续聚合

#### Scenario: CLI runtime metadata 不改变设计 recipe 身份
- **WHEN** training CLI 为已生成的 design-screen config 注入 `runtime.cli_config_path` 或其他 transient runtime metadata
- **THEN** candidate config/recipe fingerprint MUST 保持与生成 YAML 一致
- **AND** provenance 校验仍 MUST 拒绝模型、数据、profile、candidate 或 inner-split 的实际差异

### Requirement: H4 development runtime 必须隔离 outer test
带有 `mmw_t2_design_screening.development_only=true` 的训练配置 MUST 显式禁用 final test。runtime MUST 只构造和使用筛选声明的 inner train/validation splits，且产物不得包含 outer-test metrics。

#### Scenario: 执行 development-only H4 candidate
- **WHEN** trainer 接收 development-only H4 design config
- **THEN** dataloaders MUST 不含 test split，trainer MUST 跳过 final test
- **AND** final artifact MUST 将 `final_test_metrics` 记录为未执行而不是 test evidence
