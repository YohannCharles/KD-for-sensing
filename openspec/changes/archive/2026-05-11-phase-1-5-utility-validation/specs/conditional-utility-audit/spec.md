## ADDED Requirements

### Requirement: Bootstrap confidence outputs
Conditional Utility Audit consumers MUST be able to compute confidence intervals from existing per-sample audit outputs without rerunning model inference. The confidence output MUST include paired delta statistics for weak-modality additions and all-modal comparison against `strong_only`.

#### Scenario: 写出 bootstrap CI 表
- **WHEN** 系统从 `subset_predictions` 和 `conditional_utility_per_sample_delta` 计算 bootstrap confidence
- **THEN** 系统 MUST 生成 `conditional_utility_bootstrap_ci.csv` 或等价 summary 表
- **AND** 字段 MUST 至少包含 `comparison`、`weak_modality`、`metric`、`horizon_name`、`mean_delta`、`ci_lower`、`ci_upper`、`num_bootstrap`、`num_clusters` 和 `cluster_key`

#### Scenario: 复用已有 audit 输出
- **WHEN** 用户只提供已有 `outputs/scene32/<run_name>/conditional_utility/` 目录
- **THEN** 系统 MUST 能在不加载模型 checkpoint、不构建 dataloader、不执行 forward 的情况下计算 bootstrap CI
- **AND** 系统 MUST 保留输入表路径和行数 metadata

## MODIFIED Requirements

### Requirement: Conditional utility summary and diagnosis
系统 MUST 生成总 summary JSON，汇总 aggregate metrics、marginal utility、oracle、teacher complementarity、bucket highlights、metadata 和 diagnosis。Diagnosis MUST 使用配置阈值，不得硬编码在不可覆盖的位置；当 bootstrap confidence 可用时，global useful 和 conditionally useful 判定 MUST 同时满足最小效果量阈值和 CI 下界约束。

#### Scenario: 写出 summary JSON
- **WHEN** audit runner 完成所有审计步骤
- **THEN** 系统 MUST 生成 `conditional_utility_summary.json`
- **AND** summary MUST 包含 `run_name`、`scene`、`num_samples`、`horizons`、`aggregate_metrics`、`marginal_utility_vs_strong_only`、`marginal_utility_by_horizon`、`oracle_subset`、`teacher_complementarity` 和 `diagnosis`
- **AND** 当 bootstrap confidence 可用时，summary MUST 记录 CI 输出路径、bootstrap 配置、cluster key 和每个 comparison 的关键 CI

#### Scenario: 标记 global useful
- **WHEN** 某个弱模态 overall delta 达到配置的 `global_delta_dba` 或等价主指标阈值
- **AND** bootstrap confidence 可用且该主指标 CI 下界大于 0
- **THEN** diagnosis MUST 将该弱模态标记为 `globally_useful`
- **AND** diagnosis MUST 记录触发的 metric、mean delta、CI、阈值和样本/cluster 数

#### Scenario: 拒绝不显著的 tiny gain
- **WHEN** 某个弱模态 overall delta 为正但低于配置的最小效果量阈值
- **OR** bootstrap confidence 可用但该主指标 CI 跨 0
- **THEN** diagnosis MUST 不得仅凭正数 delta 将该弱模态标记为 `globally_useful`
- **AND** diagnosis MUST 将该证据记录为 `not_significant` 或等价状态

#### Scenario: 标记 conditional useful
- **WHEN** 某个弱模态 overall delta 不为正，但至少一个满足最小样本数的 bucket 达到配置的 `conditional_delta_dba` 或 `mean_delta_ce` 阈值
- **AND** bootstrap confidence 可用时对应 bucket 或 horizon 的 CI 下界大于 0
- **THEN** diagnosis MUST 将该弱模态标记为 `conditionally_useful`
- **AND** diagnosis MUST 记录触发的 bucket、horizon、指标值、CI、阈值和样本数

#### Scenario: 标记 representation exists but not exploited
- **WHEN** 某个弱模态 teacher rescue rate 达到配置阈值，但当前 MARF 的 `strong_plus_<modality>` 没有显著超过 `strong_only`
- **THEN** diagnosis MUST 将该弱模态标记为 `representation_exists_but_not_exploited`
- **AND** summary MUST 保留触发该判断的 teacher rescue 指标和 strong-plus 显著性证据
