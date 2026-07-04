## ADDED Requirements

### Requirement: Run index 二级热点必须按 scanner/collector/writer 拆分
Experiment run index 重构 MUST 拆分 output/log scanning、process/resource collection、artifact summarization、table rendering 和 JSON/CSV writing，并保持 public CLI output schema。

#### Scenario: run index 输出兼容
- **WHEN** `kd-sensing-runs` is run after refactor
- **THEN** JSON output MUST 保留 `generated_at`、`roots`、`runs`、`resources` 和 `warnings`
- **AND** default skipped output partitions MUST remain compatible
