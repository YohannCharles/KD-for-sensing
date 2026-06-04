## MODIFIED Requirements

### Requirement: 多任务训练 loss
训练和评估流程 MUST 支持 beam supervised 主损失、遮挡 BCE 和位置 MSE 的加权组合，并 MUST 输出遮挡 accuracy、blocked-class F1 和位置 RMSE。无效标签位置 MUST 被 mask，不得参与 loss 或指标。

#### Scenario: no-KD 多任务训练
- **WHEN** 用户运行启用遮挡和位置辅助任务的 fusion 训练
- **THEN** 总 loss MUST 等于 beam supervised 基础 loss 加上配置权重后的遮挡 loss 和位置 loss
- **AND** 系统 MUST 不计算 KD 基础 loss

#### Scenario: 辅助任务关闭
- **WHEN** 用户未启用遮挡或位置辅助任务
- **THEN** 训练流程 MUST 保持 beam supervised 行为
- **AND** 不得要求 auxiliary labels 存在

