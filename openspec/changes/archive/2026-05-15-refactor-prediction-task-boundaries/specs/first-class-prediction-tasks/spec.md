## ADDED Requirements

### Requirement: Objective metadata 覆盖指标和日志契约
系统 MUST 将每个 prediction objective 的默认主指标、指标方向、可用指标、early stopping alias、训练历史字段、TensorBoard 标量字段和 runtime metadata 统一定义在 objective 层。训练、验证和评估流程 MUST 消费 objective 层提供的描述，不得在 trainer 或 validator 中维护独立的 objective 指标表。

#### Scenario: occlusion objective 指标来源
- **WHEN** 配置设置 `experiment.objective: occlusion`
- **THEN** objective metadata MUST 声明默认主指标为 `val_occlusion_blocked_f1`
- **AND** objective metadata MUST 声明该指标方向为 `max`
- **AND** validation metrics 的 `available_metrics` MUST 包含 `val_occlusion_blocked_f1`
- **AND** early stopping alias `occlusion` 和 `blocked_f1` MUST 解析为 `val_occlusion_blocked_f1`

#### Scenario: position objective 指标来源
- **WHEN** 配置设置 `experiment.objective: position`
- **THEN** objective metadata MUST 声明默认主指标为 `val_position_rmse`
- **AND** objective metadata MUST 声明该指标方向为 `min`
- **AND** validation metrics 的 `available_metrics` MUST 包含 `val_position_rmse`
- **AND** early stopping alias `position` 和 `position_rmse` MUST 解析为 `val_position_rmse`

#### Scenario: 训练日志字段由 objective 声明
- **WHEN** 一次训练 epoch 完成并写入 history、epoch log 或 TensorBoard 标量
- **THEN** objective 相关字段 MUST 来自当前 objective metadata 的日志字段声明
- **AND** beam、occlusion、position 和 multitask 的既有公开字段名 MUST 继续兼容

### Requirement: Objective 可用指标校验
系统 MUST 根据当前 objective 和 validation metrics 中实际产生的标量校验 early stopping metric。用户显式配置的 metric 不在当前 objective 可用指标集合中时，系统 MUST 拒绝继续训练并报告当前可用指标。

#### Scenario: 拒绝不可用 beam 指标
- **WHEN** 配置设置 `experiment.objective: position` 且 `training.early_stopping_metric: val_adba`
- **THEN** validation 后的 early stopping 校验 MUST 失败
- **AND** 错误信息 MUST 包含 `position` objective 和可用指标列表

#### Scenario: 接受显式可用指标
- **WHEN** 配置设置 `experiment.objective: multitask` 且 `training.early_stopping_metric: val_multitask_loss`
- **THEN** early stopping 校验 MUST 通过
- **AND** checkpoint metadata MUST 记录该 primary metric 及其方向
