# first-class-prediction-tasks Specification

## Purpose
定义 beam、occlusion、position 和 multitask 预测目标的一等配置与指标契约。
## Requirements
### Requirement: 预测目标配置
系统 MUST 支持 `experiment.objective` 配置，用于选择当前实验的主预测目标。合法值 MUST 包含 `beam`、`occlusion`、`position` 和 `multitask`。当配置未提供 `experiment.objective` 时，系统 MUST 默认使用 `beam`，并保持既有训练、验证和评估行为。

#### Scenario: 默认 beam 目标
- **WHEN** 用户加载未设置 `experiment.objective` 的旧配置
- **THEN** 系统 MUST 将预测目标解析为 `beam`
- **AND** 训练、验证和评估 MUST 使用既有 beam label、beam loss、Top-K 和 DBA 指标

#### Scenario: fusion 输入与 occlusion 预测目标
- **WHEN** 配置设置 `experiment.task: fusion` 且 `experiment.objective: occlusion`
- **THEN** 系统 MUST 按 fusion 路由准备多模态输入
- **AND** 系统 MUST 按 occlusion 目标准备遮挡标签、选择遮挡 logits、计算遮挡 loss 和遮挡指标

#### Scenario: 拒绝未知预测目标
- **WHEN** 配置设置未知 `experiment.objective`
- **THEN** 系统 MUST 拒绝加载配置
- **AND** 错误信息 MUST 列出支持的预测目标

### Requirement: 预测目标 target 契约
系统 MUST 提供统一 target 准备契约，按 `experiment.objective` 返回当前目标所需的 label、valid mask 和 metadata。未被当前 objective 使用的目标 MUST 不参与主 loss。

#### Scenario: beam target
- **WHEN** `experiment.objective` 为 `beam`
- **THEN** target helper MUST 返回形状兼容 `[B, H]` 的 beam class labels
- **AND** labels MUST 使用既有 `ignore_index` 语义处理无效位置

#### Scenario: occlusion target
- **WHEN** `experiment.objective` 为 `occlusion`
- **THEN** target helper MUST 返回 `occlusion_label` 和 `occlusion_valid`
- **AND** 主 loss MUST 只使用 `occlusion_valid` 为真的位置

#### Scenario: position target
- **WHEN** `experiment.objective` 为 `position`
- **THEN** target helper MUST 返回 `position_target` 和 `position_valid`
- **AND** 主 loss MUST 只使用 `position_valid` 为真的位置

#### Scenario: multitask target
- **WHEN** `experiment.objective` 为 `multitask`
- **THEN** target helper MUST 返回 beam、occlusion 和 position 三类 targets
- **AND** 每个分任务 MUST 使用自己的 valid mask 或 ignore-index 语义

### Requirement: 预测目标 loss 契约
系统 MUST 按 `experiment.objective` 计算 primary loss。`occlusion` 和 `position` 一等任务 MUST 不依赖 `loss.alpha: 0.0` 关闭 beam loss；beam loss 只有在 `beam` 或 `multitask` objective 中默认参与总 loss。

#### Scenario: occlusion 单任务 loss
- **WHEN** `experiment.objective` 为 `occlusion`
- **THEN** 总 loss 默认 MUST 等于遮挡 BCE 主 loss
- **AND** beam CE/Focal loss MUST 不参与反向传播，除非配置显式启用诊断性附加 loss

#### Scenario: position 单任务 loss
- **WHEN** `experiment.objective` 为 `position`
- **THEN** 总 loss 默认 MUST 等于位置回归主 loss
- **AND** beam CE/Focal loss MUST 不参与反向传播，除非配置显式启用诊断性附加 loss

#### Scenario: multitask loss
- **WHEN** `experiment.objective` 为 `multitask`
- **THEN** 总 loss MUST 按配置权重合成 beam、occlusion 和 position loss
- **AND** 训练日志 MUST 分别记录每个分任务 loss 和加权总 loss

### Requirement: 预测目标指标和 early stopping
系统 MUST 为每个预测目标提供默认主指标、指标方向和 early stopping alias。用户显式配置 early stopping metric 时，系统 MUST 使用用户配置并校验该指标在当前 objective 下可用。

#### Scenario: beam 默认主指标
- **WHEN** `experiment.objective` 为 `beam`
- **THEN** 默认 early stopping metric MUST 为 `val_adba`
- **AND** 默认 mode MUST 为 `max`

#### Scenario: occlusion 默认主指标
- **WHEN** `experiment.objective` 为 `occlusion`
- **THEN** 默认 early stopping metric MUST 为 `val_occlusion_blocked_f1`
- **AND** 默认 mode MUST 为 `max`

#### Scenario: position 默认主指标
- **WHEN** `experiment.objective` 为 `position`
- **THEN** 默认 early stopping metric MUST 为 `val_position_rmse`
- **AND** 默认 mode MUST 为 `min`

#### Scenario: multitask 默认主指标
- **WHEN** `experiment.objective` 为 `multitask`
- **THEN** 系统 MUST 提供可用的默认 multitask early stopping metric
- **AND** final config MUST 记录该 metric 的组成或来源

### Requirement: 预测目标运行产物
系统 MUST 在训练产物和 final config runtime metadata 中记录解析后的 objective、主 loss 名称、主指标名称、指标方向、启用的 targets 和启用的 model heads。

#### Scenario: 单任务遮挡产物
- **WHEN** 完成 `experiment.objective: occlusion` 训练
- **THEN** `final_config.yaml` MUST 记录 objective 为 `occlusion`
- **AND** `training_outputs.npz` 或等价训练日志 MUST 包含遮挡主 loss 和遮挡主指标

#### Scenario: 单任务位置产物
- **WHEN** 完成 `experiment.objective: position` 训练
- **THEN** `final_config.yaml` MUST 记录 objective 为 `position`
- **AND** `training_outputs.npz` 或等价训练日志 MUST 包含位置主 loss 和位置主指标

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

### Requirement: Prediction objective 元数据与 loss 实现分离
系统 MUST 将 prediction objective 的元数据契约作为轻量能力暴露。元数据 MUST 覆盖 objective 名称、默认主指标、指标方向、可用指标、early stopping alias、required targets、required outputs、history fields、TensorBoard scalar 映射和 runtime metadata。该元数据 MUST 可被配置加载、配置 normalization、训练、验证和评估复用，且读取元数据时 MUST 不导入 torch loss 实现。

#### Scenario: 配置 normalization 读取 objective 元数据
- **WHEN** 配置加载流程需要根据 `experiment.objective` 补齐默认 early stopping metric 和 mode
- **THEN** 系统 MUST 使用轻量 objective 元数据
- **AND** 配置加载流程 MUST 不导入 torch、prediction loss helper 或训练 runtime

#### Scenario: 训练 runtime 复用同一 objective 元数据
- **WHEN** 训练 runtime 写出 final config、history fields、TensorBoard scalar 或 primary metric metadata
- **THEN** 系统 MUST 复用同一 objective 元数据契约
- **AND** 训练 runtime MUST 不维护与配置路径不同的 objective metric 表

#### Scenario: loss helper 保持 runtime 职责
- **WHEN** 系统计算 `beam`、`occlusion`、`position`、`multitask`、`current_beam_selection`、`current_los_classification`、`current_link_quality` 或 `selection_multitask` loss
- **THEN** torch 相关 target 和 loss 计算 MUST 位于 engine runtime 或等价重依赖模块
- **AND** 该模块 MUST 从轻量 objective 元数据读取契约，而不是复制 objective 列表和 metric 表

### Requirement: Objective 元数据公开 API 兼容
系统 MUST 保持现有 objective helper 的公开行为兼容。现有调用方如果从 `kd_sensing.engine.prediction_objectives` 读取 objective metadata helper，仍 MUST 能获得相同语义的返回值；新增轻量模块 MAY 成为内部实现来源，但不得要求用户修改训练配置或 CLI 参数。

#### Scenario: 现有 objective helper 继续可用
- **WHEN** 现有代码调用 `resolve_prediction_objective`、`objective_spec`、`objective_runtime_metadata`、`objective_history_fields` 或 `objective_tensorboard_scalars`
- **THEN** 调用 MUST 继续成功
- **AND** 返回字段和默认值 MUST 与变更前兼容

#### Scenario: 新增 objective 只改一处元数据契约
- **WHEN** 开发者新增 prediction objective 或调整 objective 默认 metric
- **THEN** 主要变更 MUST 位于轻量 objective 元数据契约及必要的 runtime loss/metric 实现
- **AND** 配置 normalization 和训练日志字段 MUST 通过该契约自动消费更新

### Requirement: Near-field beam selection objective
系统 MUST 支持 `experiment.objective: near_field_beam_selection`，用于 Multimodal-NF 当前 frame 的近场三维 codebook beam selection。该目标 MUST 与 DeepSense6G future beam prediction 区分，并 MUST 不计算 future-only DBA 或 future horizon 指标。

#### Scenario: 解析 near-field beam objective
- **WHEN** 用户加载 `experiment.objective: near_field_beam_selection`
- **THEN** 系统 MUST 将主 target 解析为当前 `target_beam`
- **AND** 系统 MUST 要求或解析 codebook metadata
- **AND** 默认 early stopping metric MUST 为 `val_beam_top1`
- **AND** 默认 metric mode MUST 为 `max`

#### Scenario: 拒绝 future horizon 语义
- **WHEN** Multimodal-NF 配置包含 future-only `num_pred > 1`、future beam horizon 或 DeepSense sequence-only target
- **THEN** 系统 MUST 拒绝该配置或将其标准化为 current frame 语义
- **AND** 错误信息 MUST 指向 `near_field_beam_selection` 当前 frame objective

### Requirement: 三维 codebook target schema
系统 MUST 支持 near-field beam target schema，用于描述三维 beam triplet、flattened class、Top-5 候选、beam power 和 codebook shape。target helper MUST 输出主训练 label，并保留结构化 metadata。

#### Scenario: 准备主 label
- **WHEN** batch 包含 `target_beam`
- **THEN** target helper MUST 返回形状兼容 `[B, 1]` 的 beam class labels
- **AND** labels MUST 表示 Top-1 三维 triplet flatten 后的 class id

#### Scenario: 准备 Top-5 metadata
- **WHEN** batch 包含 `beam_triplet_topk` 和 `beam_power_topk`
- **THEN** target helper 或 metrics payload MUST 保留这些字段用于诊断
- **AND** 主 loss MUST 默认只使用 `target_beam`
- **AND** 缺失 Top-5 metadata 时系统 MUST 能继续训练主 beam classification，并在 metadata 中记录不可用状态

### Requirement: Near-field beam loss 契约
系统 MUST 为 `near_field_beam_selection` 计算主分类 loss。默认 loss MUST 使用 flattened `target_beam` 与模型输出 beam logits；结构化 triplet loss 或 beam power weighting 只能在配置显式启用时参与。

#### Scenario: 默认分类 loss
- **WHEN** `experiment.objective` 为 `near_field_beam_selection`
- **THEN** 总 loss 默认 MUST 等于当前 beam classification 主 loss
- **AND** LoS、NF、position 或 beam power 辅助项 MUST 不参与反向传播，除非配置显式启用

#### Scenario: 输出维度校验
- **WHEN** 模型输出 beam logits 的类别数与 codebook flattened class 数不一致
- **THEN** 系统 MUST 拒绝训练或评估
- **AND** 错误信息 MUST 包含模型输出类别数、codebook shape 和期望类别数

### Requirement: Near-field beam 指标
系统 MUST 为 near-field beam selection 输出当前 frame Top-K 指标。指标 MUST 基于当前 frame label，不得使用 DeepSense future-only DBA 口径。

#### Scenario: Top-K 指标
- **WHEN** 验证或评估 near-field beam selection objective
- **THEN** metrics MUST 包含 `val_beam_top1`、`val_beam_top3` 和 `val_beam_top5`
- **AND** `available_metrics` MUST 包含这些字段和 `val_loss`
- **AND** validation metrics MUST NOT 产生 `val_adba`

#### Scenario: triplet Top-5 命中诊断
- **WHEN** batch 提供 `beam_triplet_topk`
- **THEN** metrics MAY 输出 triplet Top-5 命中或等价诊断字段
- **AND** 该诊断字段 MUST 明确标注为 current near-field codebook metric

### Requirement: Near-field objective 运行 metadata
系统 MUST 在训练产物、评估报告和 final config runtime metadata 中记录 near-field objective、codebook shape、flatten 规则、target schema、启用模态、input profiles 和辅助标签可用性。

#### Scenario: 记录 codebook metadata
- **WHEN** near-field beam selection 训练完成
- **THEN** final config 或 run metadata MUST 记录 codebook shape、codebook 文件路径或 fingerprint、flatten order 和 num beam classes
- **AND** checkpoint metadata MUST 记录 objective 为 `near_field_beam_selection`

#### Scenario: 记录辅助标签可用性
- **WHEN** Multimodal-NF dataset 暴露 `los_label`、`nf_label` 或 trajectory mode
- **THEN** run metadata MUST 记录这些辅助标签是否可用
- **AND** 如果辅助标签未参与主 loss，metadata MUST 明确其诊断或过滤用途

