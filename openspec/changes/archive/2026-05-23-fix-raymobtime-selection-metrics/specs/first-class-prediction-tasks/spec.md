## MODIFIED Requirements

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

#### Scenario: Raymobtime selection 单任务 TensorBoard 隔离
- **WHEN** 当前 objective 为 `current_beam_selection`
- **THEN** objective metadata 的 TensorBoard scalar MUST 包含 `beam/val_top1`、`beam/val_top3`、`beam/val_top5` 和 `beam/val_dba_current`
- **AND** 该 objective 的 TensorBoard scalar MUST NOT 包含 `los/accuracy`、`los/f1`、`los/auc`、`link/mae`、`link/rmse` 或 `link/r2`

#### Scenario: Raymobtime LOS 单任务 TensorBoard 隔离
- **WHEN** 当前 objective 为 `current_los_classification`
- **THEN** objective metadata 的 TensorBoard scalar MUST 包含 `los/accuracy`、`los/f1` 和 `los/auc`
- **AND** 该 objective 的 TensorBoard scalar MUST NOT 包含 `beam/val_top1`、`beam/val_top3`、`beam/val_top5`、`beam/val_dba_current`、`link/mae`、`link/rmse` 或 `link/r2`

#### Scenario: Raymobtime link 单任务 TensorBoard 隔离
- **WHEN** 当前 objective 为 `current_link_quality`
- **THEN** objective metadata 的 TensorBoard scalar MUST 包含 `link/mae`、`link/rmse` 和 `link/r2`
- **AND** 该 objective 的 TensorBoard scalar MUST NOT 包含 `beam/val_top1`、`beam/val_top3`、`beam/val_top5`、`beam/val_dba_current`、`los/accuracy`、`los/f1` 或 `los/auc`

### Requirement: Current beam selection 预测目标
系统 MUST 支持 `experiment.objective: current_beam_selection`，用于当前 snapshot 的最优 beam class 分类。该目标 MUST 与既有 future beam prediction 区分，并 MUST 不计算 future-only DBA 指标。系统 MUST 为该目标提供当前 beam 距离敏感指标，命名为 `val_beam_dba`，不得复用 legacy `val_adba`。

#### Scenario: 解析 current beam selection objective
- **WHEN** 用户加载 `experiment.objective: current_beam_selection` 的配置
- **THEN** 系统 MUST 将主 target 解析为当前 `target_beam`
- **AND** 系统 MUST 将主模型输出解析为当前 beam logits
- **AND** 默认 early stopping metric MUST 为 `val_beam_top1`
- **AND** 默认 early stopping mode MUST 为 `max`

#### Scenario: current beam target 形状
- **WHEN** batch 包含 Raymobtime s008 当前 beam label
- **THEN** target helper MUST 返回形状兼容 `[B, 1]` 的 beam class labels
- **AND** 该 label MUST 表示当前最优 Tx/Rx beam pair 的 class

#### Scenario: current beam 指标
- **WHEN** 验证或评估 current beam selection objective
- **THEN** validation metrics MUST 包含 `val_beam_top1`、`val_beam_top3`、`val_beam_top5` 和 `val_beam_dba`
- **AND** `available_metrics` MUST 包含 `val_beam_top1`、`val_beam_top3`、`val_beam_top5`、`val_beam_dba` 和 `val_loss`
- **AND** validation metrics MUST 不要求且 MUST NOT 产生 `val_adba`

#### Scenario: current beam DBA alias
- **WHEN** 用户为 current beam selection 配置 `training.early_stopping_metric: beam_dba`、`current_beam_dba`、`val_beam_dba` 或 `beam/val_dba_current`
- **THEN** objective metadata MUST 将该 alias 解析为 `val_beam_dba`
- **AND** metric mode MUST 为 `max`

### Requirement: Selection multitask 预测目标
系统 MUST 支持 `experiment.objective: selection_multitask`，用于同时训练当前 beam selection、LOS/NLOS 分类和 link quality 回归。该目标 MUST 使用独立的 target、loss、metric 和日志字段，不得复用既有 occlusion/position 多任务语义。

#### Scenario: 解析 selection multitask objective
- **WHEN** 用户加载 `experiment.objective: selection_multitask` 的配置
- **THEN** 系统 MUST 启用 `beam_selection`、`los` 和 `link_quality` 三类 targets
- **AND** 系统 MUST 要求模型输出 beam logits、LOS logits 和 link prediction
- **AND** 默认 early stopping metric MUST 为 `val_selection_multitask_loss`
- **AND** 默认 early stopping mode MUST 为 `min`

#### Scenario: selection multitask target 准备
- **WHEN** batch 包含 `target_beam`、`los_label` 和 `link_quality`
- **THEN** target helper MUST 返回 beam class label、LOS 二值 label 和 link regression target
- **AND** 返回 target MUST 与当前 snapshot 单步输出对齐
- **AND** 缺失任一必需 target 时 MUST 报出包含 objective 名称和缺失字段的清晰错误

#### Scenario: selection multitask loss 合成
- **WHEN** selection multitask objective 计算训练 loss
- **THEN** 总 loss MUST 等于配置权重后的 beam CE、LOS BCEWithLogits 和 link SmoothL1 分量之和
- **AND** 训练日志 MUST 分别记录 `loss/beam_selection`、`loss/los`、`loss/link_quality` 和 `loss/selection_multitask_total`

#### Scenario: selection multitask 指标
- **WHEN** 验证或评估 selection multitask objective
- **THEN** metrics MUST 包含 beam Top-K、`val_beam_dba`、LOS accuracy/F1/AUC、link MAE/RMSE/R2 和 `val_selection_multitask_loss`
- **AND** `available_metrics` MUST 包含这些字段，以支持 early stopping 校验和结果汇总
- **AND** TensorBoard scalar MUST 同时包含 beam、LOS、link 和 selection multitask total loss 的正式 tag

### Requirement: Raymobtime 单任务 LOS 与 link quality 预测目标
系统 MUST 支持 Raymobtime s008 的 LOS/NLOS 分类和 link quality 回归作为独立单任务 objective。单任务 objective MUST 复用当前 snapshot batch 契约，并 MUST 不要求 future-only DBA 指标。单任务 objective 的正式 metrics、history 和 TensorBoard 输出 MUST 只暴露当前 objective 的指标。

#### Scenario: 解析 current LOS classification objective
- **WHEN** 用户加载 `experiment.objective: current_los_classification` 的 Raymobtime s008 配置
- **THEN** 系统 MUST 将主 target 解析为 `los_label`
- **AND** 系统 MUST 将主模型输出解析为 `los_logits`
- **AND** 默认 early stopping metric MUST 为 `val_los_f1`
- **AND** 默认 early stopping mode MUST 为 `max`

#### Scenario: 解析 current link quality objective
- **WHEN** 用户加载 `experiment.objective: current_link_quality` 的 Raymobtime s008 配置
- **THEN** 系统 MUST 将主 target 解析为 `link_quality`
- **AND** 系统 MUST 将主模型输出解析为 `link_quality`
- **AND** 默认 early stopping metric MUST 为 `val_link_mae`
- **AND** 默认 early stopping mode MUST 为 `min`

#### Scenario: LOS 单任务指标
- **WHEN** 验证或评估 `current_los_classification`
- **THEN** validation metrics MUST 输出 `val_los_accuracy`、`val_los_f1` 和 `val_los_auc`
- **AND** `available_metrics` MUST 只暴露 `val_loss` 和 LOS 单任务指标
- **AND** validation metrics MUST NOT 暴露 `val_beam_top1`、`val_beam_top3`、`val_beam_top5`、`val_beam_dba`、`val_link_mae`、`val_link_rmse` 或 `val_link_r2`

#### Scenario: link 单任务指标
- **WHEN** 验证或评估 `current_link_quality`
- **THEN** validation metrics MUST 输出 `val_link_mae`、`val_link_rmse` 和 `val_link_r2`
- **AND** `available_metrics` MUST 只暴露 `val_loss` 和 link 单任务指标
- **AND** validation metrics MUST NOT 暴露 `val_beam_top1`、`val_beam_top3`、`val_beam_top5`、`val_beam_dba`、`val_los_accuracy`、`val_los_f1` 或 `val_los_auc`
