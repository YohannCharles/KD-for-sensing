## ADDED Requirements

### Requirement: Eval matrix aggregation fields
U-MaskBeamJEPA eval matrix MUST 输出 seed aggregation 和 strict comparability 所需字段。新增字段 MUST 不改变既有模型 forward、训练行为或默认 fixed/random pattern 语义。

#### Scenario: eval row 包含 comparability fields
- **WHEN** eval matrix 写出 pattern-level CSV/JSON
- **THEN** 每行 MUST 包含或在 metadata 中关联 run_name、method、seed、split、sample_count、label_space、metric_profile、target_source、modalities、pattern_name 和 difficulty_digest
- **AND** 缺失字段 MUST 用 warning 或空值表达，不得静默伪造

#### Scenario: pattern group summary
- **WHEN** eval matrix 完成多个 missing pattern
- **THEN** 系统 SHOULD 输出或支持派生 full、avg_missing、overall_mean、balanced、only_gps 和 non_gps_only 等 group metrics
- **AND** group 定义 MUST 写入 JSON metadata，供统计模块和 claim harvester 复用

#### Scenario: stress suite 复用 eval matrix
- **WHEN** missing-modality stress suite 调用 eval matrix
- **THEN** eval matrix MUST 接收显式 missing mask 或 difficulty-transformed batch
- **AND** 原 batch 的 target、beam_power、sample id 和 split metadata MUST 不被原地修改
