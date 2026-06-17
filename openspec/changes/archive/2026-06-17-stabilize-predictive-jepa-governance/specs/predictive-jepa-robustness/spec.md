## ADDED Requirements

### Requirement: Predictive Robustness 文档治理
Predictive Robustness 作为 current capability 时，系统 MUST 在 spec Purpose、lifecycle inventory、主线模型目录、实验协议表和 claim 账本中明确它是 pending/unverified 的 current workflow capability，而不是已经完成真实数值 claim 的结果。

#### Scenario: current capability 但 claim 未验证
- **WHEN** 文档登记 `predictive-jepa-robustness` 为 current capability
- **THEN** 文档 MUST 明确真实 claim 仍需要 strict comparable train-then-evaluate run
- **AND** synthetic metrics、mock weights、partial model set 或 allow_missing_artifacts MUST 只能标记为 `mock/smoke`、`pending` 或 `unavailable`

#### Scenario: spec Purpose 描述真实能力边界
- **WHEN** 维护者打开 `openspec/specs/predictive-jepa-robustness/spec.md`
- **THEN** `## Purpose` MUST 说明 Predictive Robustness 用于评估 JEPA 预测表征在当前图像不可观测或 GPS plausibly-wrong 条件下的鲁棒性
- **AND** Purpose MUST 不包含 `TBD`、归档 scaffold 文案或未验证数值 claim

### Requirement: 训练 profile 与完整 P-suite benchmark 分离
Predictive Robustness MUST 区分训练 difficulty profile、evaluation difficulty profile 和 P0-P5 benchmark suite。单个训练 profile 或 clean evaluation profile MUST NOT 被描述为完整 P0-P5 regional benchmark。

#### Scenario: 训练配置只启用部分 predictive condition
- **WHEN** 派生训练配置只声明 `P4_joint_predictive_recovery` 或其它单个 predictive condition
- **THEN** 文档 MUST 将其描述为训练/curriculum profile
- **AND** 文档 MUST 指向 benchmark manifest 或本地 real manifest 才能执行完整 P0-P5 regional evaluation

#### Scenario: 完整 benchmark claim 需要 P-suite provenance
- **WHEN** claim registry 准备将 Predictive Robustness 升级为真实 claim
- **THEN** provenance MUST 包含 P0-P5 condition-level metrics、strict comparability fields、difficulty digest、seed、split、sample_count 和 CNN+GPS baseline
- **AND** 缺少任一 required provenance 时 claim status MUST 保持 `pending`、`unavailable` 或 `not_comparable`
