## REMOVED Requirements

### Requirement: Phase 1.5 运行清单
**Reason**: Phase 1.5 Utility Validation 属于已放弃的弱模态效用验证流程。
**Migration**: 删除 manifest 和入口；普通实验矩阵应通过各自配置或新的 OpenSpec change 定义。

#### Scenario: Phase 1.5 manifest 退役
- **WHEN** 用户查找 Phase 1.5 清单配置
- **THEN** 系统不再要求提供 `configs/analysis/phase_1_5_utility_validation.yaml`

### Requirement: Bootstrap 显著性分析
**Reason**: bootstrap CI 只服务于 Phase 1.5 弱模态效用判断。
**Migration**: 不提供迁移；新的统计验证需重新提出。

#### Scenario: Bootstrap 分析退役
- **WHEN** 用户运行分析工具
- **THEN** 系统不再要求输出 `conditional_utility_bootstrap_ci.csv`

### Requirement: MARF checkpoint matrix 复核
**Reason**: checkpoint matrix 依赖 Conditional Utility Audit 和 Phase 1.5 判定。
**Migration**: 使用普通 checkpoint 评估流程对单个 checkpoint 复评估。

#### Scenario: checkpoint matrix 退役
- **WHEN** 用户查看 Phase 1.5 输出
- **THEN** 系统不再要求生成 checkpoint audit commands 或 comparison CSV

### Requirement: Dedicated fixed-subset baseline 矩阵
**Reason**: fixed-subset baseline matrix 是 Phase 1.5 弱模态效用验证的一部分。
**Migration**: 若需要新的 baseline matrix，应在实验矩阵文档或新 change 中定义。

#### Scenario: fixed-subset baseline matrix 退役
- **WHEN** 用户运行分析工具
- **THEN** 系统不再要求生成 Phase 1.5 baseline 命令或 summary

### Requirement: Phase 1.5 决策报告
**Reason**: Phase 1.5 路线决策标签不再代表项目方向。
**Migration**: 使用当前研究目标对应的 metrics/report。

#### Scenario: Phase 1.5 report 退役
- **WHEN** 用户查看分析输出
- **THEN** 系统不再要求生成 `phase_1_5_summary.json` 或 `phase_1_5_report.md`

### Requirement: Phase 1.5 非侵入性
**Reason**: Phase 1.5 本体已退役，其非侵入性约束不再需要单独维护。
**Migration**: 普通训练和评估继续由通用 workflow specs 约束。

#### Scenario: Phase 1.5 非侵入性约束退役
- **WHEN** 用户运行普通训练或评估
- **THEN** 系统不再需要检查 Phase 1.5 是否启用
