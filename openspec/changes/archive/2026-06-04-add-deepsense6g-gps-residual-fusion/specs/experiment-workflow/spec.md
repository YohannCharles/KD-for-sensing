## ADDED Requirements

### Requirement: Residual workflow query leakage guard
DeepSense6G residual workflow MUST prevent target query labels from being used for prior construction, support selection, early stopping, model selection or hyperparameter tuning.

#### Scenario: query label 只用于最终评价
- **WHEN** 系统运行 `target_adapt_beambench_residual`
- **THEN** target query label MUST only be used to compute final metrics, predictions diagnostics, figures and comparison report
- **AND** run metadata MUST record `query_label_used_for_training=false`
- **AND** model selection split MUST be source validation or target support internal validation

#### Scenario: support/query role 可审计
- **WHEN** residual manifest 和 predictions 被写出
- **THEN** 每一行 MUST 包含 support/query role
- **AND** summary MUST record support count and query count for each target scene

### Requirement: Residual workflow result contract
DeepSense6G residual workflow MUST produce machine-readable summaries that can compare every residual ablation against GPS v2 baseline.

#### Scenario: residual summary 字段
- **WHEN** residual evaluation 完成
- **THEN** summary MUST include protocol、support ratio、label space、train mode、ablation、modalities、num samples、DBA、DBA zero ratio、mean/median circular error、exact/pm/top-k metrics、GPS baseline metrics and residual deltas
- **AND** 所有 error 字段 MUST 使用 circular distance

#### Scenario: gps_prior_only 复现 GPS v2
- **WHEN** 用户运行 `gps_prior_only`
- **THEN** 系统 MUST 读取 GPS v2 r15 prior predictions
- **AND** summary MUST numerically match v2 r15 within documented tolerance
- **AND** 若不能匹配 MUST 在 comparison report 中记录差异原因

#### Scenario: 推荐方法选择规则
- **WHEN** comparison report 选择推荐 residual method
- **THEN** 系统 MUST 首先考虑 overall DBA
- **AND** 系统 MUST 同时检查 good sample degradation rate
- **AND** DBA 略高但大量破坏 GPS good 样本的方法 MUST NOT 被标为推荐方法

### Requirement: Residual workflow acceptance commands
实现完成后 residual workflow MUST 提供可在 `kd_mm_beam` 环境中运行的 inspection、manifest、train/eval、plot、compare 和 test 命令。

#### Scenario: 验收命令使用 kd_mm_beam
- **WHEN** 开发者运行 residual workflow 验收
- **THEN** 所有 Python 命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** 命令 MUST 使用 `kd_sensing` 包内 CLI 或 console script
- **AND** 验收 MUST 包含 residual 新测试与现有 circular metrics 回归
