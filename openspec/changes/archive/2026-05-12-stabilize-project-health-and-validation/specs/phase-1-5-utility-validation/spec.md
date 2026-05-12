## MODIFIED Requirements

### Requirement: Phase 1.5 运行清单
系统 MUST 提供 Phase 1.5 Utility Validation 运行清单，用于声明输入 audit 目录、MARF checkpoint matrix、dedicated fixed-subset baseline matrix、随机种子、输出目录和判定阈值。运行清单 MUST 可复现地记录每个产物来自哪个配置、checkpoint、seed 和 subset。系统 MUST 将缺失或未完成的关键产物标记为 `missing` 或 `pending`，并且这些缺失项 MUST 阻止 Phase 1.5 总决策进入 final `complete` 状态。

#### Scenario: 读取 Phase 1.5 清单
- **WHEN** 用户启动 Phase 1.5 汇总入口并传入运行清单
- **THEN** 系统 MUST 解析 `conditional_utility` 输入目录、checkpoint roles、baseline subset、seed 列表和输出目录
- **AND** 系统 MUST 在输出 metadata 中保存解析后的清单内容

#### Scenario: 缺失关键产物
- **WHEN** 运行清单引用的 audit 表、checkpoint 或 baseline metrics 不存在
- **THEN** 系统 MUST 将对应条目标记为 `missing` 或 `pending`
- **AND** 系统 MUST 不把缺失条目纳入 final decision gate
- **AND** 总 `decision.status` MUST 保持 `pending`
- **AND** 总 `decision.evidence_level` MUST 保持 `exploratory`

#### Scenario: checkpoint matrix 未完成时保持 pending
- **WHEN** bootstrap CI 和 dedicated baseline matrix 均已完成
- **AND** checkpoint matrix 中任一 role 的 checkpoint、audit summary 或必要输入仍为 `missing` 或 `pending`
- **THEN** Phase 1.5 summary MUST 将 `checkpoint_matrix.status` 标记为 `pending`
- **AND** Phase 1.5 summary MUST 将总 `decision.status` 标记为 `pending`
- **AND** 系统 MAY 输出 bootstrap 和 baseline 的局部判断作为探索性证据

### Requirement: Phase 1.5 决策报告
系统 MUST 生成 Phase 1.5 总报告，汇总 bootstrap CI、checkpoint matrix、dedicated baseline matrix、bucket highlights、teacher complementarity、oracle gain 和最终路线建议。报告 MUST 明确区分最终证据和探索性证据。只有 bootstrap、checkpoint matrix 和 dedicated baseline matrix 都达到完成状态时，报告 MUST 输出 final 证据级别和最终路线标签。

#### Scenario: 弱模态无稳定收益
- **WHEN** dedicated strong+weak 与 all baseline 在 3 seeds 下没有显著超过 dedicated `gps+mmwave`
- **AND** bootstrap CI 不支持 Phase 1 masking delta 为正
- **AND** checkpoint matrix 已完成且不支持弱模态稳定正收益
- **THEN** 报告 MUST 将 Scene32 clean setting 标记为 `low_weak_utility`
- **AND** 报告 MUST 推荐后续转向 strong-path 精度、safe fusion 和 RF/GPS degraded robustness，而不是 MARF-Comm
- **AND** 报告 MUST 将 `decision.evidence_level` 标记为 `final`

#### Scenario: 弱模态存在条件性收益
- **WHEN** 某个弱模态在 dedicated baseline 或 checkpoint matrix 中于特定 bucket/horizon 上稳定正收益
- **AND** 对应 CI 下界大于 0 且样本数满足配置阈值
- **AND** bootstrap、checkpoint matrix 和 dedicated baseline matrix 均已完成
- **THEN** 报告 MUST 将该弱模态标记为 `conditionally_useful`
- **AND** 报告 MUST 允许后续进入 MARF-Comm 条件效用 router 设计
- **AND** 报告 MUST 将 `decision.evidence_level` 标记为 `final`

#### Scenario: 关键矩阵未完成时仅输出探索性报告
- **WHEN** bootstrap、checkpoint matrix 或 dedicated baseline matrix 任一状态不是 `complete`
- **THEN** 报告 MUST 将总 `decision.status` 标记为 `pending`
- **AND** 报告 MUST 将 `decision.label` 标记为 `pending`
- **AND** 报告 MUST 将 `decision.evidence_level` 标记为 `exploratory`
- **AND** 报告 MUST 保留已完成矩阵的局部输出路径和状态，供用户继续补齐产物

#### Scenario: 报告保留实际数据出处
- **WHEN** Phase 1.5 报告引用任意数值结论
- **THEN** 报告 MUST 记录该数值来自 audit table、checkpoint summary、baseline metrics 或 bootstrap output
- **AND** 报告 MUST 保存输入文件路径、run name、checkpoint role、seed 和 subset
