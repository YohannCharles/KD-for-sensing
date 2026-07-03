## ADDED Requirements

### Requirement: Run index claim-harvester fields
实验运行索引 MUST 提供 claim harvester 可消费的稳定字段，但 MUST 不承担 claim 判定规则库。新增字段 MUST 保持只读，并且缺失时以 warning 或空值表达。

#### Scenario: run summary 包含 identity 和 artifact paths
- **WHEN** run index 扫描到一个训练或评估 run
- **THEN** run summary MUST 包含 run_name、run_dir、config_path、config_digest、seed、scene_scope、dataset_family、metric_profile、target_source 和 artifact path 摘要
- **AND** 如果字段无法解析，summary MUST 保留 run 基本状态并记录 warning

#### Scenario: run summary 包含 eval artifacts
- **WHEN** run index 扫描到 evaluation 或 missing-pattern 输出
- **THEN** summary MUST 记录 eval artifact 类型、CSV/JSON path、mtime、size 和关联 run_name
- **AND** run index MUST 不解析 claim readiness 或统计显著性

#### Scenario: 当前进程关联 run
- **WHEN** run index 发现当前训练进程
- **THEN** summary SHOULD 记录 config path、run name、PID、GPU index 和 command line
- **AND** dashboard MAY 使用这些字段展示 running 状态
