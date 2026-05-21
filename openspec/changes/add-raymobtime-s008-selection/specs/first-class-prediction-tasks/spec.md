## ADDED Requirements

### Requirement: Current beam selection 预测目标
系统 MUST 支持 `experiment.objective: current_beam_selection`，用于当前 snapshot 的最优 beam class 分类。该目标 MUST 与既有 future beam prediction 区分，并 MUST 不计算 future-only DBA 指标。

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
- **THEN** validation metrics MUST 包含 `val_beam_top1`、`val_beam_top3` 和 `val_beam_top5`
- **AND** validation metrics MUST 不要求 `val_adba`

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
- **THEN** metrics MUST 包含 beam Top-K、LOS accuracy/F1/AUC、link MAE/RMSE/R2 和 `val_selection_multitask_loss`
- **AND** `available_metrics` MUST 包含这些字段，以支持 early stopping 校验和结果汇总

### Requirement: Raymobtime 单任务 LOS 与 link quality 预测目标
系统 MUST 支持 Raymobtime s008 的 LOS/NLOS 分类和 link quality 回归作为独立单任务 objective。单任务 objective MUST 复用当前 snapshot batch 契约，并 MUST 不要求 future-only DBA 指标。

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

#### Scenario: 单任务指标
- **WHEN** 验证或评估 `current_los_classification` 或 `current_link_quality`
- **THEN** LOS 单任务 MUST 输出 LOS accuracy/F1/AUC
- **AND** link 单任务 MUST 输出 link MAE/RMSE/R2
- **AND** `available_metrics` MUST 只暴露当前 objective 可用于 early stopping 的 Raymobtime 单任务指标

### Requirement: Selection objective 运行产物
系统 MUST 在 Raymobtime current beam selection 和 selection multitask 的训练产物、评估报告和 final config runtime metadata 中记录 objective、启用 tasks、主 metric、metric mode、target 口径和 model heads。

#### Scenario: current beam selection 产物
- **WHEN** current beam selection 训练完成
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录 `objective: current_beam_selection`
- **AND** 运行产物 MUST 记录 `task_semantics: current_snapshot_beam_selection`
- **AND** 运行产物 MUST 记录 beam class 数、Tx/Rx beam 数和当前 snapshot split 信息

#### Scenario: selection multitask 产物
- **WHEN** selection multitask 训练完成
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录 `objective: selection_multitask`
- **AND** 运行产物 MUST 记录启用的 `beam_selection`、`los` 和 `link_quality` heads
- **AND** 运行产物 MUST 记录 loss 权重、link target 名称和 LOS label 来源

#### Scenario: 拒绝 future 命名配置
- **WHEN** Raymobtime s008 配置包含 `future_beam`、`beam_prediction_horizon`、`beam_tracking` 或 LOS transition 目标
- **THEN** 配置校验 MUST 拒绝该配置
- **AND** 错误信息 MUST 指向 `current_beam_selection` 或 `selection_multitask` objective
