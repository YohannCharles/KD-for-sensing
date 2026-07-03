## ADDED Requirements

### Requirement: Multi-seed statistical summary
系统 MUST 支持对缺失模态实验结果进行多 seed 统计汇总。汇总 MUST 按 method、seed、pattern、metric 和可选 family 分组，并输出 mean、std、count、delta 和置信区间。

#### Scenario: 汇总 method seed pattern
- **WHEN** 输入包含多个 method、seed 和 missing pattern 的 metrics rows
- **THEN** 系统 MUST 输出 method-level summary
- **AND** summary MUST 至少包含 method、metric、seed_count、pattern_count、mean、std、min、max 和 source artifact paths

#### Scenario: bootstrap 置信区间
- **WHEN** 用户启用 bootstrap 或使用默认统计配置
- **THEN** 系统 MUST 输出每个核心指标的 confidence interval
- **AND** 输出 MUST 记录 bootstrap seed、sample count、iteration count 和 confidence level

#### Scenario: 样本不足降级
- **WHEN** 某 method 只有一个 seed 或样本不足以计算 std/CI
- **THEN** 系统 MUST 保留该 method 的点估计
- **AND** std、CI 或显著性字段 MUST 标记为 unavailable，并记录 warning

### Requirement: Paired comparison evidence
系统 MUST 支持同 seed 或同 run-pair 的 paired comparison，用于比较候选方法和 baseline。无法配对时 MUST 不伪造 paired 证据。

#### Scenario: paired delta 计算
- **WHEN** candidate 和 baseline 在同一 seed、split、pattern、metric profile 和 label space 下都有结果
- **THEN** 系统 MUST 计算 paired delta、win/loss/tie count 和 per-pattern delta
- **AND** 输出 MUST 记录 baseline method 和 pairing keys

#### Scenario: paired keys 不完整
- **WHEN** candidate 和 baseline 的 seed、split、pattern 或 metric profile 无法配对
- **THEN** 系统 MUST 标记 paired comparison 为 unavailable
- **AND** warning MUST 说明缺失或冲突的 pairing keys

### Requirement: Claim-oriented statistical gate
统计模块 MUST 提供 claim-oriented gate，用于判定结果是否具备进入 claim draft 的统计证据。Gate MUST 不自动修改正式 claim registry。

#### Scenario: 满足 claim gate
- **WHEN** candidate 相对 baseline 的 primary metric delta 为正、seed_count 达到配置阈值、comparability 为 strict 且 CI 不跨过配置的最小效果阈值
- **THEN** 系统 MAY 标记 `statistical_claim_ready=true`
- **AND** 输出 MUST 记录阈值、baseline、primary metric 和 caveat

#### Scenario: 不满足 claim gate
- **WHEN** seed_count 不足、CI 跨阈值、paired evidence unavailable 或 comparability 不是 strict
- **THEN** 系统 MUST 标记 `statistical_claim_ready=false`
- **AND** 输出 MUST 给出 next action，例如补 seed、补 fresh eval 或修复 comparability field
