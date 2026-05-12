# phase-1-5-utility-validation Specification

## Purpose
TBD - created by archiving change phase-1-5-utility-validation. Update Purpose after archive.
## Requirements
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

### Requirement: Bootstrap 显著性分析
系统 MUST 基于 Phase 1 逐样本表计算 paired bootstrap 置信区间。Bootstrap MUST 覆盖 `strong_plus_image - strong_only`、`strong_plus_radar - strong_only`、`strong_plus_lidar - strong_only` 和 `all - strong_only`，指标 MUST 包含 Top1、Top3、DBA 和 CE，并 MUST 输出 overall 与 per-horizon 结果。

#### Scenario: 计算弱模态边际 CI
- **WHEN** `conditional_utility_per_sample_delta` 包含 `delta_top1`、`delta_top3`、`delta_dba` 和 `delta_ce`
- **THEN** 系统 MUST 为每个弱模态计算均值差、95% CI、bootstrap sample count 和正向比例
- **AND** 系统 MUST 按 `horizon_name` 分别输出同一组统计量

#### Scenario: 计算 all-modal CI
- **WHEN** `subset_predictions` 同时包含 `all` 和 `strong_only`
- **THEN** 系统 MUST 按相同样本与 horizon 配对计算 `all - strong_only` 的 Top1、Top3、DBA 和 CE delta
- **AND** 系统 MUST 将结果写入与弱模态边际 CI 相同的表结构

#### Scenario: cluster key fallback
- **WHEN** 逐样本表包含 `seq_id`
- **THEN** bootstrap MUST 按 `seq_id` cluster 重采样
- **AND** 当 `seq_id` 不存在时，系统 MUST 使用 `sample_id` 或 `dataset_index` 作为 fallback cluster key，并在输出 metadata 中记录 fallback 类型

### Requirement: MARF checkpoint matrix 复核
系统 MUST 支持对同一 MARF run 的多个 checkpoint role 运行或汇总 Conditional Utility Audit。默认 checkpoint role MUST 至少包含 `best_top1`、`best` 和 `last`；如果存在显式 `best_dba` checkpoint，系统 MUST 能记录并纳入比较。

#### Scenario: 生成 checkpoint audit 矩阵
- **WHEN** 运行清单指定 `scene32_marf` 的 checkpoint roles
- **THEN** 系统 MUST 为每个 role 解析 checkpoint 路径和输出 audit 目录
- **AND** 系统 MUST 复用现有 audit runner 生成缺失的单 checkpoint audit 产物

#### Scenario: 汇总 checkpoint 结论
- **WHEN** 多个 checkpoint role 的 audit 产物可用
- **THEN** 系统 MUST 输出每个 role 的 subset metrics、marginal utility、oracle gain、teacher complementarity 和 diagnosis
- **AND** 系统 MUST 标记弱模态结论是否跨 checkpoint 一致

### Requirement: Dedicated fixed-subset baseline 矩阵
系统 MUST 定义并汇总 dedicated fixed-subset baseline 训练矩阵。矩阵 MUST 包含 `gps+mmwave`、`gps+mmwave+image`、`gps+mmwave+radar`、`gps+mmwave+lidar` 和 `image+radar+gps+lidar+mmwave` 五个 subset，且每个 subset MUST 至少运行 3 个 seed。

#### Scenario: 生成 baseline 命令
- **WHEN** 用户请求生成 Phase 1.5 baseline 命令
- **THEN** 系统 MUST 为五个 fixed-subset 和配置的 seed 列表生成训练命令
- **AND** 每条命令 MUST 使用相同训练预算、checkpoint 选择规则、encoder 初始化策略、loss 配置和评估协议

#### Scenario: 汇总 baseline 指标
- **WHEN** baseline metrics 对五个 subset 和所有 seed 均可用
- **THEN** 系统 MUST 输出每个 subset 的 Top1、Top3、DBA 和 loss 的 `mean ± std`
- **AND** 系统 MUST 同时输出 `t+1`、`t+2`、`t+3` 和 average 指标

#### Scenario: dedicated strong-only 作为主基线
- **WHEN** 系统比较 strong+weak 或 all baseline
- **THEN** 系统 MUST 使用同一 seed matrix 下的 dedicated `gps+mmwave` 作为主基线
- **AND** 当前 MARF masking `strong_only` 指标只能作为 Phase 1 诊断参照，不得替代 dedicated 主基线

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

### Requirement: Phase 1.5 非侵入性
Phase 1.5 MUST 不改变普通训练、普通评估、MARF 模型结构、router 输入、loss 配置、encoder 冻结策略或已有 Phase 1 audit 默认行为。所有新增计算 MUST 只在显式 Phase 1.5 入口或配置中触发。

#### Scenario: 普通训练不触发 Phase 1.5
- **WHEN** 用户运行现有训练配置且未启用 Phase 1.5
- **THEN** 系统 MUST 不生成 bootstrap CI、checkpoint matrix、baseline matrix 或 Phase 1.5 report
- **AND** 系统 MUST 保持现有训练输出语义

#### Scenario: 普通 audit 不触发 baseline 训练
- **WHEN** 用户只运行 Conditional Utility Audit
- **THEN** 系统 MUST 不自动启动 dedicated fixed-subset baseline 训练
- **AND** 系统 MUST 只生成该 audit 入口定义的单 checkpoint 诊断产物
