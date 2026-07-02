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

#### Scenario: current beam selection 单任务 TensorBoard 隔离
- **WHEN** 当前 objective 为 `current_beam_selection`
- **THEN** objective metadata 的 TensorBoard scalar MUST 包含 `beam/val_top1`、`beam/val_top3`、`beam/val_top5` 和 `beam/val_dba_current`
- **AND** 该 objective 的 TensorBoard scalar MUST NOT 包含 `los/accuracy`、`los/f1`、`los/auc`、`link/mae`、`link/rmse` 或 `link/r2`

#### Scenario: current LOS 单任务 TensorBoard 隔离
- **WHEN** 当前 objective 为 `current_los_classification`
- **THEN** objective metadata 的 TensorBoard scalar MUST 包含 `los/accuracy`、`los/f1` 和 `los/auc`
- **AND** 该 objective 的 TensorBoard scalar MUST NOT 包含 `beam/val_top1`、`beam/val_top3`、`beam/val_top5`、`beam/val_dba_current`、`link/mae`、`link/rmse` 或 `link/r2`

#### Scenario: current link quality 单任务 TensorBoard 隔离
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
- **WHEN** batch 包含当前 snapshot beam label
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

### Requirement: current snapshot LOS 与 link quality 预测目标
系统 MUST 支持当前 snapshot 的 LOS/NLOS 分类和 link quality 回归作为独立单任务 objective。单任务 objective MUST 复用当前 snapshot batch 契约，并 MUST 不要求 future-only DBA 指标。单任务 objective 的正式 metrics、history 和 TensorBoard 输出 MUST 只暴露当前 objective 的指标。Raymobtime s008 dataset/config 本身已退役，不能作为这些 objective 的当前数据入口。

#### Scenario: 解析 current LOS classification objective
- **WHEN** 用户加载 `experiment.objective: current_los_classification` 的当前 snapshot 配置
- **THEN** 系统 MUST 将主 target 解析为 `los_label`
- **AND** 系统 MUST 将主模型输出解析为 `los_logits`
- **AND** 默认 early stopping metric MUST 为 `val_los_f1`
- **AND** 默认 early stopping mode MUST 为 `max`

#### Scenario: 解析 current link quality objective
- **WHEN** 用户加载 `experiment.objective: current_link_quality` 的当前 snapshot 配置
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
系统 MUST 在 current beam selection 和 selection multitask 的训练产物、评估报告和 final config runtime metadata 中记录 objective、启用 tasks、主 metric、metric mode、target 口径和 model heads。

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
- **WHEN** current snapshot objective 配置包含 `future_beam`、`beam_prediction_horizon`、`beam_tracking` 或 LOS transition 目标
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

### Requirement: Selection multitask target metadata
`selection_multitask` objective MUST 在 runtime metadata 中记录 beam selection、LOS classification 和 link quality regression 三个 target 的启用状态、loss 字段和 metric 字段。

#### Scenario: Multitask metadata 完整
- **WHEN** 当前 objective 为 `selection_multitask`
- **THEN** runtime metadata MUST 记录启用 targets、主 metric、metric mode、每个 target 的 output/head 名称和关键 loss 字段
- **AND** metrics output MUST 能追溯 beam、LOS 和 link 三类指标

### Requirement: Objective metadata 可合并但行为不可变
预测目标的默认 metric、metric mode、available metrics、alias、history fields、TensorBoard scalars 和 runtime metadata MAY 合并到单一 owner 表或 helper。合并 MUST 保持每个 objective 的公开字段名和校验行为不变。

#### Scenario: objective metric alias 保持兼容
- **WHEN** 用户配置 beam、occlusion、position、multitask、current beam、LOS、link、selection multitask 或 JEPA objective 的 early stopping alias
- **THEN** alias MUST 解析为与变更前相同的 canonical metric
- **AND** metric mode MUST 保持一致

#### Scenario: TensorBoard scalar 保持隔离
- **WHEN** 当前 objective 为 beam、current beam selection、LOS、link、selection multitask 或 JEPA
- **THEN** objective metadata MUST 输出该 objective 对应的 TensorBoard scalar 集合
- **AND** 不属于该 objective 的 scalar MUST 不被错误写入

### Requirement: Objective 小表不拆成伪 registry
Objective metadata 合并 MUST 不新增 registry、factory、adapter 或多文件常量拆分来替代现有小表。新增 objective 才需要通过新的 OpenSpec change 扩展 metadata 表。

#### Scenario: 合并 history 和 registry 常量
- **WHEN** `_DEFAULT_METRICS`、metric aliases、history fields 或 TensorBoard scalar 表被合并
- **THEN** 调用方 MUST 继续通过 objective metadata owner 查询
- **AND** trainer、validator 和 tensorboard logging MUST 不维护独立重复表

### Requirement: 未来标签时隙对齐
训练、验证、评估、诊断预测导出和 KD 相关 loss MUST 使用 `num_pred` 个未来标签时隙。`num_pred=3` 时，系统 MUST 将 label 和预测 slot 解释为 `[t+1, t+2, t+3]`，不得包含当前或历史最后一个 beam。

#### Scenario: 训练 loss 使用未来标签
- **WHEN** 训练流程准备 batch 且 `num_pred: 3`
- **THEN** loss 输入 logits MUST 与 `[t+1, t+2, t+3]` 三个标签时隙对齐
- **AND** flatten 后的 logits 数量 MUST 等于 flatten 后的 labels 数量

#### Scenario: 输出 slot 选择使用 future horizon
- **WHEN** 模型输出 logits 的时间维长度大于或等于 `num_pred`
- **THEN** 统一 slot 选择 helper MUST 返回最后 `num_pred` 个 slot
- **AND** 返回结果 MUST 与 `prepare_labels()` 输出的 future labels 同长
- **AND** 该 helper 的语义 MUST 表示长时序输出对齐，不得作为方法专属额外 prediction slot 的兼容承诺

#### Scenario: 输出 slot 不足时报错
- **WHEN** 模型输出 logits 的时间维长度小于 `num_pred`
- **THEN** 训练、验证或评估流程 MUST 报出清晰错误
- **AND** 系统 MUST 不通过重复、padding 或拼接历史 beam 自动补齐 prediction slots

#### Scenario: 诊断预测导出保留 t+1
- **WHEN** viewer prediction export 写出 `confidence_curves` 或 `beam_distribution`
- **THEN** 导出的第一个 horizon MUST 表示 `t+1`
- **AND** 导出逻辑 MUST 不把第一个预测 slot 当作 current beam 丢弃

### Requirement: Future horizon flat metrics
验证和评估输出 MUST 在现有 nested top-k 数组之外，增加 future horizon 扁平指标字段。字段 MUST 使用 `t1/t2/t3/avg` 命名，并 MUST 不输出历史 current beam 或 h0 指标。

#### Scenario: 保存三步 Top-K 扁平字段
- **WHEN** 验证阶段产出 logits `[B,3,64]` 和 labels `[B,3]`
- **THEN** `metrics.json` MUST 包含 `val_top1_t1`、`val_top1_t2`、`val_top1_t3` 和 `val_top1_avg`
- **AND** `metrics.json` MUST 包含 `val_top3_avg` 和 `val_top5_avg`
- **AND** 这些 avg 字段 MUST 对有效 future horizon 求平均

#### Scenario: 不输出旧 h0 指标
- **WHEN** 普通 future-only 评估写出 metrics
- **THEN** metrics MUST 不包含 `top1_h0`
- **AND** metrics MUST 不包含 `top1_future_avg`
- **AND** metrics MUST 不包含 `beam8_acc`

### Requirement: Metric horizon aggregation consistency
训练验证、force-mask subset 验证和 standalone evaluate MUST 对 beam Top-K、ADBA/DBA 和公开 top-level scalar 使用同一套 selected metric horizons。配置或 runtime 解析出的 `metric_horizons` MUST 被记录在 metrics metadata 中，subset top-level scalar MUST NOT 回退到 first valid slot 口径。

#### Scenario: subset top1 使用 selected horizons
- **WHEN** 配置选择 `metric_horizons=[2,4,6]` 或等价 horizon 集合
- **THEN** 普通 validation 的 top-level Top-1 MUST 基于这些 selected horizons 聚合
- **AND** force-mask subset validation 的 top-level `top1` 或等价 scalar MUST 使用同一 selected horizon 聚合
- **AND** subset validation MUST NOT 使用 first valid slot 作为 top-level `top1`

#### Scenario: standalone evaluate 记录同一口径
- **WHEN** 用户通过 standalone evaluate 运行同一配置
- **THEN** evaluate metrics/report MUST 记录实际使用的 `metric_horizons`
- **AND** Top-K 与 DBA/ADBA top-level scalar MUST 与训练验证使用同一 horizon 选择规则
- **AND** 若输出逐 horizon 诊断，诊断字段 MUST 与 top-level 聚合字段可区分

#### Scenario: 未配置 horizons 使用统一默认
- **WHEN** 配置没有显式设置 `metric_horizons`
- **THEN** validation、subset validation 和 evaluate MUST 使用同一个默认 horizon 集合
- **AND** metrics metadata MUST 记录默认来源或等价说明

### Requirement: Objective metrics 可用性语义
训练、验证和评估流程 MUST 区分 active objective metrics 与 inactive metrics。未启用、缺少 head、缺少 target 或未实际计算的任务指标 MUST 不被写成 `0.0` 真实性能；系统 MUST 用缺失、`null`、`NaN` 或显式 availability metadata 表示不可用状态。

#### Scenario: beam-only 训练不写 position 零曲线
- **WHEN** 用户运行 `experiment.objective: beam` 且未启用 position target/head 的训练
- **THEN** TensorBoard MUST 不写入 `position/rmse` 或 `position/mae` 标量曲线
- **AND** epoch log MUST 不把 `val_position_rmse` 或 `val_position_mae` 记录为真实 `0.0`

#### Scenario: occlusion-only 训练不写 position 零曲线
- **WHEN** 用户运行 `experiment.objective: occlusion` 且未启用 position target/head 的训练
- **THEN** TensorBoard MUST 不写入 `position/rmse` 或 `position/mae` 标量曲线
- **AND** `training_outputs.npz` 若保留 position metric 数组 key，inactive slot MUST 使用 `NaN` 或等价不可用表示

#### Scenario: position-only 训练不写 occlusion 零曲线
- **WHEN** 用户运行 `experiment.objective: position` 且未启用 occlusion target/head 的训练
- **THEN** TensorBoard MUST 不写入 `occlusion/accuracy` 或 `occlusion/blocked_f1` 标量曲线
- **AND** epoch log MUST 不把 `val_occlusion_accuracy` 或 `val_occlusion_blocked_f1` 记录为真实 `0.0`

#### Scenario: multitask 训练写入全部 active metrics
- **WHEN** 用户运行 `experiment.objective: multitask` 且 beam、occlusion 和 position metrics 均可计算
- **THEN** TensorBoard MUST 写入 beam、occlusion 和 position 对应的 active scalar 曲线
- **AND** `train_log.json` MUST 记录三个任务的验证指标和 multitask 加权总 loss

#### Scenario: early stopping 不接受 inactive metric
- **WHEN** 用户配置的 early stopping metric 对当前 objective 不可用
- **THEN** 训练流程 MUST 在保存 misleading checkpoint 前抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的 metric，并提示用户选择当前 objective 可用的 metric

### Requirement: Objective-aware validation 输出
验证和评估输出 MUST 只把真实计算的 auxiliary metrics 提升为 top-level metric，并 MUST 提供足够 metadata 说明哪些 objective targets 和 heads 已启用。inactive metric 不得通过默认零值绕过下游 early stopping 和图表解释。

#### Scenario: metrics JSON 省略 inactive auxiliary metric
- **WHEN** 验证 `experiment.objective: beam` 且未计算 position metric
- **THEN** `metrics.json` MUST 不把 top-level `val_position_rmse` 写成 `0.0`
- **AND** 输出 MUST 能通过 objective metadata 表明 position 不是本次 enabled head/target

#### Scenario: metrics JSON 包含 active position metric
- **WHEN** 验证 `experiment.objective: position` 且 position output、target 和 valid mask 均可用
- **THEN** `metrics.json` MUST 包含真实计算的 `val_position_rmse`
- **AND** 该值 MUST 用 position target scaler 反归一化后的尺度计算

#### Scenario: metrics JSON 包含 active occlusion metric
- **WHEN** 验证 `experiment.objective: occlusion` 且 occlusion logits、label 和 valid mask 均可用
- **THEN** `metrics.json` MUST 包含真实计算的 `val_occlusion_blocked_f1`
- **AND** 该值 MUST 可作为 `val_occlusion_blocked_f1/max` early stopping 来源
