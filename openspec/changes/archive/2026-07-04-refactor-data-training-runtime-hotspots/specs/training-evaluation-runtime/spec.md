## ADDED Requirements

### Requirement: Training 与 evaluation runtime 必须保持阶段边界
Training 和 evaluation runtime 重构 MUST 保持 context preparation、resource construction、state restore、epoch loop、evaluation step、metric aggregation 和 finalization 这些显式阶段，并保持公开行为稳定。

#### Scenario: training context 拆分
- **WHEN** training context preparation or resource construction is refactored
- **THEN** run directory creation, initial config artifacts, normalization artifacts, startup summary, AMP, non-blocking transfer and resume validation MUST remain compatible

#### Scenario: evaluation pass 拆分
- **WHEN** evaluation pass internals are split
- **THEN** `EvaluationPassResult`, objective metadata, prediction metadata, metric keys and difficulty stage scoping MUST remain compatible
