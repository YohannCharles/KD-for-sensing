## ADDED Requirements

### Requirement: H2R 简化筛选保持单因素可归因
系统 SHALL 使用同一 source checkpoint、Joint panel、seed、batch、mask identity 和评估 cache 运行固定八候选，只允许 evidence profile、Router 辅助损失和校准 epoch 按预注册矩阵变化。

#### Scenario: 生成八卡计划
- **WHEN** 用户生成完整 H2R 简化筛选计划
- **THEN** manifest 精确包含 GPU0--7 各一个预注册候选
- **AND** 每项记录 profile、supervision、loss 权重、epoch、checkpoint SHA 和 panel SHA

### Requirement: 筛选结果保持开发证据边界
系统 MUST 将训练配置、checkpoint、日志和 Joint 评估写入 ignored `outputs/`，并将所有候选标记为 seed1 inner-only、不可形成正式 claim。

#### Scenario: 候选完成训练与评估
- **WHEN** 任一候选生成 checkpoint 和 Joint summary
- **THEN** provenance 标明固定 inner split、seed1 和 `claim_eligible=false`
- **AND** canonical T2/S1 配置及正式 claim 文档不被修改

