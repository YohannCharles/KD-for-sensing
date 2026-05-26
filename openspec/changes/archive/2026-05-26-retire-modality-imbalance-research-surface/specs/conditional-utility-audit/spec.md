## REMOVED Requirements

### Requirement: Conditional audit subset registry
**Reason**: Conditional Utility Audit 研究流程已退役，项目不再维护其专属 subset registry 或 `strong_plus_*` 审计命名契约。
**Migration**: 需要普通模态子集评估时使用现有 `evaluation.modality_subsets` 与模型的 `force_modality_mask` 能力。

#### Scenario: Conditional audit subset registry 退役
- **WHEN** 用户查找 Conditional Utility Audit 专属 subset registry
- **THEN** 系统不再要求提供该 registry 或其 metadata 输出

### Requirement: Conditional audit runner
**Reason**: Conditional Utility Audit 入口和配置已退役。
**Migration**: 使用普通 `kd-sensing-evaluate`、`scripts/eval_modality_subsets.py` 或针对具体实验的新 change 重新定义需要的评估产物。

#### Scenario: Conditional audit runner 退役
- **WHEN** 用户查找 `tools/analysis/run_conditional_utility_audit.py`
- **THEN** 系统不再要求提供该入口或 `conditional_utility/` 输出目录

### Requirement: Per-sample subset predictions
**Reason**: 逐样本 subset prediction dump 是 Conditional Utility Audit 的内部产物，研究线退役后不再维护。
**Migration**: 如需新的逐样本导出，应在新的分析 capability 中重新定义字段和入口。

#### Scenario: subset prediction dump 退役
- **WHEN** 普通训练或评估完成
- **THEN** 系统不再要求生成 `subset_predictions` 审计表

### Requirement: Marginal utility deltas
**Reason**: 弱模态边际效用统计已不再作为项目研究目标。
**Migration**: 不提供迁移；后续若重新需要边际效用分析，应新建独立 OpenSpec change。

#### Scenario: marginal delta 退役
- **WHEN** 普通训练或评估完成
- **THEN** 系统不再要求生成 `conditional_utility_per_sample_delta`

### Requirement: Teacher complementarity audit
**Reason**: teacher rescue/complementarity 统计属于已放弃的弱模态解释流程。
**Migration**: G2D teacher ensemble 仅保留训练用途，不再作为 Conditional Utility Audit 的 teacher dump 入口。

#### Scenario: teacher complementarity audit 退役
- **WHEN** 用户启用普通 G2D 或 no-KD 训练
- **THEN** 系统不再要求输出 `teacher_predictions` 或 `teacher_complementarity_summary.json`

### Requirement: Subset oracle
**Reason**: subset oracle 是 Conditional Utility 研究判断的一部分，已退役。
**Migration**: 不提供迁移；新的 oracle 分析必须重新定义 capability。

#### Scenario: subset oracle 退役
- **WHEN** 普通评估运行
- **THEN** 系统不再要求生成 `oracle_subset_summary.json`

### Requirement: Communication state buckets
**Reason**: 通信状态 bucket 下的弱模态效用统计已不再维护。
**Migration**: 保留普通 viewer diagnostics；如需新的 bucket 统计，应针对当前任务重新设计。

#### Scenario: communication bucket audit 退役
- **WHEN** 普通评估运行
- **THEN** 系统不再要求生成 `conditional_utility_by_bucket.csv`

### Requirement: Conditional utility summary and diagnosis
**Reason**: summary diagnosis 使用弱模态 useful/conditional useful 等研究标签，已与当前项目方向不符。
**Migration**: 使用普通 `metrics.json`、`test_report.json` 和任务特定报告作为结果依据。

#### Scenario: conditional utility diagnosis 退役
- **WHEN** 用户运行普通训练或评估
- **THEN** 系统不再要求输出 `conditional_utility_summary.json` 或弱模态效用 diagnosis

### Requirement: Audit visualizations
**Reason**: Conditional Utility figures 依赖已退役的审计表。
**Migration**: 保留现有 manifest viewer 与普通可视化流程。

#### Scenario: audit figures 退役
- **WHEN** 用户查看分析工具
- **THEN** 系统不再要求提供 `tools/analysis/analyze_conditional_utility.py`

### Requirement: Non-invasive audit behavior
**Reason**: Conditional Utility Audit 本体已退役，对其非侵入性约束不再需要单独维护。
**Migration**: 普通训练和评估仍由现有 workflow specs 约束。

#### Scenario: audit 非侵入性约束退役
- **WHEN** 用户运行普通训练或评估
- **THEN** 系统不再需要区分是否启用 Conditional Utility Audit

### Requirement: Bootstrap confidence outputs
**Reason**: bootstrap confidence 输出服务于 Phase 1/1.5 弱模态效用判断，已退役。
**Migration**: 不提供迁移；新的置信区间分析必须重新提出规格。

#### Scenario: bootstrap confidence 输出退役
- **WHEN** 用户只保留历史 audit 输出目录
- **THEN** 系统不再要求从这些表中计算 `conditional_utility_bootstrap_ci.csv`
