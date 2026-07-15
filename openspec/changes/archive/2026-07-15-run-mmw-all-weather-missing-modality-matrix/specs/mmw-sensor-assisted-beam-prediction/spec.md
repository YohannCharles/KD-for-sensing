## ADDED Requirements

### Requirement: All-weather sensor-assisted comparison eligibility
MMW all-weather 主比较 MUST 对 S1、T2、AMBER-Full 与 RMBP-MM 使用相同的 image、GPS、LiDAR、radar sensing input inventory，并 MUST 将 CSI、channel、mmWave、beam-power、path 和 radio label 保持为非输入字段。任何 domain 缺少启用 sensing modality 的 run MUST 标记 `main_conclusion_eligible=false`。

#### Scenario: 四方法输入边界一致
- **WHEN** 四方法 all-weather config 解析完成
- **THEN** 每个 config 的 enabled sensing modalities MUST 等于 image、radar、GPS、LiDAR
- **AND** sensitive usage flags MUST 证明 CSI、channel、mmWave、beam-power、path 和 radio label 未用于模型输入或 validation supervision

### Requirement: Weather and scene macro reporting
All-weather summary MUST 同时报告每个 condition/scenario 的绝对指标、相对 matched baseline delta、weather macro、scene macro、15-domain macro 和 worst-domain，并 MUST 保留 missing pattern 与 temporal rate 维度。

#### Scenario: 样本量不均衡
- **WHEN** 不同 domain 的 validation sample counts 不相等
- **THEN** 15-domain macro MUST 先计算每个 domain 指标再等权平均
- **AND** summary MUST 同时记录 micro average 但不得用它替代主 domain macro

