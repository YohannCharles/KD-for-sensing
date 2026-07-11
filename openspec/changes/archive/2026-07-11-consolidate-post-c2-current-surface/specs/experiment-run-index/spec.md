## ADDED Requirements

### Requirement: Run index consumer-neutral provenance fields
实验运行索引 MUST 保留 cleanup、人工 claim 审阅和故障排查可消费的稳定 identity/artifact 字段，但 MUST 不承担 claim 判定或 dashboard 展示逻辑。字段缺失时 MUST 以 warning 或空值表达。

#### Scenario: Run summary 包含 identity 和 artifact paths
- **WHEN** run index 扫描训练或评估 run
- **THEN** summary MUST 包含 run_name、run_dir、config path/digest、seed、scene scope、dataset family、metric profile、target source 和 artifact path 摘要
- **AND** 无法解析字段时 MUST 保留基本状态并记录 warning

#### Scenario: Run summary 包含 eval artifacts
- **WHEN** run index 扫描 evaluation 或 missing-pattern 输出
- **THEN** summary MUST 记录 artifact 类型、path、mtime、size 和关联 run name
- **AND** run index MUST 不解析 claim readiness 或生成 dashboard next action

## REMOVED Requirements

### Requirement: Run index claim-harvester fields
**Reason**: Research claim harvester/dashboard 退役，run index 不应保留专属下游命名；进程/GPU 字段与 cleanup 活跃运行保护仍保留。
**Migration**: 使用新增的 consumer-neutral provenance requirement；正式 claim 判断留在 registry/protocol。

#### Scenario: Run index 不依赖 harvester
- **WHEN** run index 构建 summary
- **THEN** 它 MUST 不导入 research claim harvester 或 dashboard
- **AND** cleanup 与人工审阅所需基础字段仍保留
