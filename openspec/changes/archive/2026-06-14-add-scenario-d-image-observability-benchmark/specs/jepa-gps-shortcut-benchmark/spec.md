## ADDED Requirements

### Requirement: Scenario D benchmark suite
JEPA GPS shortcut benchmark MUST 支持 Scenario D image observability suite。Suite MUST 复用 shared difficulty pipeline，且 MUST 能与 existing Scenario C async GPS suite 组合为 Cx-Dy matrix。

#### Scenario: manifest 引用 Scenario D suite
- **WHEN** benchmark manifest 声明 suite type `scenario_d_image_observability`
- **THEN** runner MUST 标准化 D-level condition、image operator 参数、seed 和 output artifact plan
- **AND** runner MUST 将 image corruption 委托给 shared difficulty operator
- **AND** runner MUST 不维护独立平行的 image corruption 实现

#### Scenario: Scenario C 与 D 联合执行
- **WHEN** manifest 声明 joint suite `scenario_c_x_d_image_observability`
- **THEN** runner MUST 对每个模型执行 Scenario C condition 与 Scenario D condition 的笛卡尔组合
- **AND** 每个 row MUST 记录 `gps_condition`、`image_condition`、C severity、D severity、seed 和 difficulty digest

### Requirement: Scenario D required model groups
Benchmark MUST 支持 Scenario D 指定的模型组：GPS-only、CNN+GPS、Image-AE+GPS、Image-JEPA only 和 Image-JEPA+GPS。Runner MUST 将这些模型组映射到现有 config/weights/registry 语义，并 MUST 记录模型是否消费 image/GPS reliability metadata。

#### Scenario: required model group 校验
- **WHEN** manifest 声明 strict Scenario D evaluation
- **THEN** runner MUST 校验 required model groups 是否齐全，或在显式允许 partial run 时记录缺失模型组
- **AND** report MUST 区分 standard fusion、CNN/AE visual encoder、JEPA visual encoder 和 observability-aware fusion

#### Scenario: Image-JEPA only 不消费 GPS 输入
- **WHEN** model group 为 Image-JEPA only
- **THEN** runner MUST 仍按 Cx-Dy 条件记录 GPS condition metadata 以保持矩阵对齐
- **AND** 模型 forward MUST 不要求 GPS input tensor

### Requirement: Scenario D aggregation 和图表
Benchmark MUST 聚合 Scenario D matrix，并导出 Cx-Dy heatmap、robustness surface、phase transition、CNN vs JEPA crossing point 和 modality dominance 图表或表格。图表生成失败时，metrics CSV 和 manifest MUST 仍然写出，并记录 warning。

#### Scenario: 输出 Cx-Dy aggregation
- **WHEN** Scenario D matrix 完成至少一个模型
- **THEN** runner MUST 写出包含 model、gps_condition、image_condition、metric、sample_count、seed 和 clean delta 的 long-form CSV
- **AND** runner MUST 写出按模型排序的 heatmap NPY 或等价矩阵 artifact

#### Scenario: attention 不可用时 dominance 降级
- **WHEN** 某个模型不提供 attention 或 fusion weights
- **THEN** modality dominance ratio MUST 使用配置声明的 fallback 或跳过该模型
- **AND** warnings MUST 记录 unavailable reason
