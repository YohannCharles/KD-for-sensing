## ADDED Requirements

### Requirement: Scene meta-offset 辅助目标契约
系统 MUST 支持 scene meta-offset 训练和评估中的 beam power、angle、LOS/NLOS、path count、dominant path angle 或等价 radio/geometry 字段作为 auxiliary target、loss 或 diagnostic。辅助目标 MUST 通过 objective/loss metadata 和 target provider 显式声明，并 MUST NOT 成为测试输入或 target adaptation oracle。

#### Scenario: beam power 只作为辅助 target
- **WHEN** batch 包含 `beam_power` 或 normalized beam power distribution
- **THEN** 配置启用 beam power 辅助损失时，loss helper MUST 只在允许的 labeled/source subset 上计算 KL、MSE 或 NRP proxy 辅助损失
- **AND** model input mapping MUST NOT 将真实 query/test `beam_power` 作为 sensing input 传给模型

#### Scenario: angle label 只作为辅助监督
- **WHEN** batch 包含 angle/AoD/AoA label
- **THEN** angle loss MUST 只在允许的 labeled/source subset 上参与训练
- **AND** 配置启用 angle 辅助评估时，evaluation MUST 使用 angle prediction 写出辅助指标
- **AND** query/test 真实 angle label MUST NOT 用于生成 scene embedding、offset 参数或模型输入

#### Scenario: radio path 字段防泄漏
- **WHEN** batch 包含 LOS/NLOS、path count、dominant path angle、path descriptor 或 radio semantic label
- **THEN** 这些字段 MUST 被标记为 auxiliary target 或 diagnostic
- **AND** target_unlabeled、label_budget=0 和 target_test 使用这些字段训练、调参或选择阈值 MUST 失败

#### Scenario: loss 日志分项记录
- **WHEN** scene meta-offset 配置启用 CE、ordinal、beam power、angle、aux、offset regularization、smoothness 或 gate regularization loss
- **THEN** training history MUST 分别记录各分量和加权总 loss
- **AND** final config/runtime metadata MUST 记录每个启用 loss 的权重、目标字段和 split/subset policy
