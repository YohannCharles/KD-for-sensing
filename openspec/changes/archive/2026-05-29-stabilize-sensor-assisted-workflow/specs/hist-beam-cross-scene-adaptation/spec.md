## ADDED Requirements

### Requirement: Quick validation conclusion 排除不可用于主结论的 run
HiST-Beam quick validation conclusion MUST 消费 run-level eligibility metadata。`main_conclusion_eligible=false`、target leakage、未授权 target sensitive supervision、prototype no-op 或关键对比 run 缺失的结果 MUST 不被描述为主结论改进。

#### Scenario: ineligible run 不参与胜负判断
- **WHEN** 同一 fold、budget 和 seed 下某个 adapter 或 prototype run 记录 `main_conclusion_eligible=false`
- **THEN** quick validation conclusion MUST 不把该 run 用于证明方法优于 source-only 或 full fine-tuning
- **AND** conclusion MUST 记录该 run 被排除的 variant、target scene、budget、seed 和 eligibility reasons

#### Scenario: excluded baseline 导致比较不可判定
- **WHEN** 生成 adapter/prototype 与 source-only 或 full fine-tuning 对比所需的 baseline run 缺失或被标记为不可用于主结论
- **THEN** 对应比较 MUST 标记为 `inconclusive`
- **AND** conclusion MUST 记录缺失或被排除的 run key 和原因

#### Scenario: prototype no-op 不作为有效 prototype 证据
- **WHEN** prototype run 的 metrics 标记 prototype status 为 `no_op`、`unavailable`、coverage 为 0 或 prototype loss 未实际生效
- **THEN** conclusion MUST 不把该 run 描述为有效 prototype variant
- **AND** 若 accuracy 仍有变化，conclusion MUST 将变化归为补充诊断而不是 prototype 主结论

#### Scenario: conclusion 汇总 eligibility
- **WHEN** quick validation conclusion 文件写出
- **THEN** 文件 MUST 包含 eligible run 数、excluded run 数、inconclusive comparison 数和 exclusion reason histogram
- **AND** 文件 MUST 引用产生 eligibility metadata 的 summary 或 run artifact 路径
