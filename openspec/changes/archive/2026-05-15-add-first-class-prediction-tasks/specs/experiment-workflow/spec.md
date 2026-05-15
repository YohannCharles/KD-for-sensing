## ADDED Requirements

### Requirement: Objective-aware 训练流程
训练流程 MUST 根据 `experiment.objective` 选择主 target、主模型输出、主 loss 和训练日志字段。`experiment.task` MUST 继续决定输入路由和模型 forward 路径。

#### Scenario: fusion occlusion 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: occlusion` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用遮挡 logits 和遮挡标签计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion position 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: position` 的训练配置
- **THEN** trainer MUST 使用 fusion 输入准备逻辑运行 student model
- **AND** trainer MUST 使用位置输出和位置目标计算主 loss
- **AND** trainer MUST 不要求 beam loss 参与总 loss

#### Scenario: fusion multitask 训练
- **WHEN** 用户运行 `experiment.task: fusion` 且 `experiment.objective: multitask` 的训练配置
- **THEN** trainer MUST 计算 beam、occlusion 和 position 三个 loss 分量
- **AND** trainer MUST 按配置权重合成总 loss

### Requirement: Objective-aware 验证和评估
验证和评估流程 MUST 根据 `experiment.objective` 输出当前目标的主 metrics，并保留可计算的诊断 metrics。主 metrics MUST 支持 checkpoint 选择和 standalone evaluate。

#### Scenario: occlusion 验证指标
- **WHEN** 验证 `experiment.objective: occlusion` 的模型
- **THEN** validator MUST 输出遮挡 loss、accuracy 和 blocked-class F1
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_occlusion_blocked_f1`

#### Scenario: position 验证指标
- **WHEN** 验证 `experiment.objective: position` 的模型
- **THEN** validator MUST 输出位置 loss、RMSE 和 MAE
- **AND** epoch log MUST 暴露可用于 early stopping 的 `val_position_rmse`

#### Scenario: multitask 验证指标
- **WHEN** 验证 `experiment.objective: multitask` 的模型
- **THEN** validator MUST 输出 beam、occlusion 和 position 的分任务 metrics
- **AND** validator MUST 输出 multitask 总 loss 或配置指定的主指标

### Requirement: Objective-aware checkpoint registry
checkpoint registry 和 final config MUST 记录 objective-aware 指标，确保后续 evaluation 能按训练目标解释 checkpoint。

#### Scenario: 归档 occlusion checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: occlusion` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和遮挡指标
- **AND** evaluate MUST 能读取 registry artifact 并复用遮挡阈值

#### Scenario: 归档 position checkpoint
- **WHEN** 训练完成并归档 `experiment.objective: position` 的最佳 checkpoint
- **THEN** registry metadata MUST 记录 objective、best metric、metric mode 和位置指标
- **AND** evaluate MUST 能读取 registry artifact 并复用位置 target scaler
