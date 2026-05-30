## ADDED Requirements

### Requirement: Sensor-assisted 主结论 eligibility
MMW sensor-assisted run MUST 在 run metadata 和 summary 中明确记录是否可用于主结论。任何使用不允许 sensing input、target_test 训练信息、未授权 target sensitive supervision 或不符合 sensor-assisted profile 的 run MUST 失败或标记 `main_conclusion_eligible=false`，并记录机器可读原因。

#### Scenario: source auxiliary 不影响主结论 eligibility
- **WHEN** sensor-assisted run 只在 source split 使用 beam_power、radio semantic label、path descriptor 或 path semantic label 构造 source auxiliary head、prototype 或离线 diagnostics
- **THEN** run MUST NOT 仅因 source auxiliary/prototype 使用这些字段而被标记为不可用于主结论
- **AND** metadata MUST 记录这些字段未作为 target training supervision 使用
- **AND** sensing input modality 列表 MUST 仍不包含 mmWave、CSI/channel、beam_power、path 或 radio label

#### Scenario: target radio supervision 排除主结论
- **WHEN** sensor-assisted target adaptation 使用 labeled target `radio_semantic_label` 计算训练 loss
- **THEN** run MUST 记录 `used_target_radio_label_for_training=true`
- **AND** run MUST 标记 `main_conclusion_eligible=false`，除非对应实验规格明确允许该 target supervision 进入主结论
- **AND** summary MUST 将该 run 归入补充或诊断结果，而不是主结论比较集合

#### Scenario: target path supervision 排除主结论
- **WHEN** sensor-assisted target adaptation 使用 labeled target `path_semantic_label`、`path_descriptor` 或 `path_params` 计算训练 loss
- **THEN** run MUST 记录对应 `used_target_path_*_for_training` flag
- **AND** run MUST 标记 `main_conclusion_eligible=false`，除非对应实验规格明确允许该 target supervision 进入主结论
- **AND** summary MUST 记录 exclusion reason，不能只输出 accuracy 数值

#### Scenario: summary 写出 eligibility 字段
- **WHEN** MMW sensor-assisted summary 写出
- **THEN** 每个 run record MUST 包含 `main_conclusion_eligible`、`eligibility_reasons`、enabled sensing modalities、excluded sensitive fields 和 sensitive usage flags
- **AND** summary MUST 提供可过滤主结论 run 与补充/诊断 run 的机器可读字段
