## ADDED Requirements

### Requirement: MMW quick validation run eligibility audit
MMW sensor-assisted quick validation MUST audit run eligibility using split leakage diagnostics and actual target-side oracle usage. A run MUST NOT be marked ineligible only because optional path/radio supervision fields exist in the dataset when those fields were not consumed by adaptation, model selection, threshold selection, temperature fitting, prototype update or training loss.

#### Scenario: 无 oracle 的 sensor-assisted run 可进入主结论
- **WHEN** quick validation run 使用 strict-validation eligible split，且 adaptation 只读取 sensing inputs 与允许的 target_adapt labeled support beam labels
- **THEN** run metadata MUST 标记该 run 可用于 sensor-assisted 主结论
- **AND** summary MUST NOT 因数据集中存在 path/radio/channel 文件而把该 run 排除
- **AND** metadata MUST 记录 `used_target_oracle_fields=[]` 或等价空列表

#### Scenario: 使用禁用 target oracle 的 run 被排除
- **WHEN** run 在 adaptation、threshold selection、temperature fitting、prototype update、early stopping 或 loss 计算中读取 target_test label、target_test beam_power、target_test path fields、target-side radio/channel labels 或禁用 oracle 字段
- **THEN** run metadata MUST 标记该 run 不可用于主结论
- **AND** summary MUST 将该 run 归入 excluded/debug results
- **AND** exclusion reason MUST 包含实际使用的字段、使用阶段和机器可读 reason code

#### Scenario: eligibility reason 可审计
- **WHEN** quick validation summary 计算 eligible run count
- **THEN** summary MUST 为每个 excluded run 记录 `eligibility_status`、`eligibility_reasons`、split diagnostics path 和 oracle usage summary
- **AND** `eligible_run_count=0` 时 summary MUST 能解释是 split 不严格、oracle 使用违规、产物缺失还是 validator 条件无法判定

### Requirement: MMW target oracle usage metadata
MMW quick validation run artifacts MUST 记录 target-side field usage，使 sensor-assisted、mmWave-assisted、path/radio-assisted 和 debug run 可以被机器过滤。该记录 MUST 覆盖配置声明与运行时实际消费字段。

#### Scenario: 记录允许与禁用字段
- **WHEN** run 启动或完成 adaptation/evaluation
- **THEN** run metadata MUST 记录 enabled sensing modalities、excluded target oracle fields、allowed target labels 和实际 consumed fields
- **AND** metadata MUST 区分 `target_adapt_labeled_support_label`、`target_adapt_unlabeled_input`、`target_test_evaluation_label` 与禁用 target-side path/radio/channel 字段

#### Scenario: target_test 标签只用于最终评价
- **WHEN** target_test label 被用于计算最终 Top-K、within3、MAE、histogram、KL 或 confusion artifact
- **THEN** metadata MUST 将该用途标记为 `evaluation_only`
- **AND** eligibility checker MUST NOT 因 evaluation-only target_test label 使用而排除 run
- **AND** 该标签 MUST NOT 出现在 adaptation optimizer、prior 初始化、prototype update、threshold selection、temperature fitting 或 early stopping 记录中

#### Scenario: 运行时字段消费未知时保守排除
- **WHEN** run metadata 无法证明 target-side oracle 字段未被 adaptation 或选择逻辑消费
- **THEN** eligibility checker MUST 将该 run 标记为 `unknown_oracle_usage`
- **AND** summary MUST 将该 run 排除出主结论
- **AND** exclusion reason MUST 指向缺失的 metadata 或审计字段

### Requirement: MMW v9 quick validation protocol metadata
MMW quick validation protocol MUST 能声明 v9 input-conditioned target adaptation 实验矩阵，并 MUST 将 eligibility、collapse diagnostics 和 prototype diagnostics 接入 summary。

#### Scenario: v9 quick validation plan metadata
- **WHEN** planner builds v9 quick validation plan
- **THEN** plan metadata MUST include profile、source scenes、target scene、budgets、seeds、v9 group ids 和每个 mode 的实验目的
- **AND** 默认 v9 quick validation MUST 限定为小矩阵，不得静默扩展到完整 budget/seed sweep

#### Scenario: v9 summary 汇总 collapse diagnostics
- **WHEN** v9 run 完成 source-only 或 adapted target_test evaluation
- **THEN** summary MUST 引用 `prediction_hist.json` 和 `collapse_diagnostics.json` 或等价 artifact path
- **AND** summary MUST 汇总 Top-K、within3、MAE、unique predicted beams、pred top beams、histogram KL 和 beta/prototype diagnostics

#### Scenario: Group C 需要 protocol 许可
- **WHEN** v9 Group C 使用未标注 target_adapt 做 distribution regularization 或 consistency loss
- **THEN** plan/run metadata MUST 证明未标注样本来自 target_adapt 且不包含 target_test
- **AND** metadata MUST 证明 target_test label、beam_power、path fields 和 radio labels 未参与训练或选择
- **AND** 若证明缺失，Group C run MUST 被标记为 disabled、debug 或 ineligible
