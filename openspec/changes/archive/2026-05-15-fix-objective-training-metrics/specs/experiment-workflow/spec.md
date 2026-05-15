## MODIFIED Requirements

### Requirement: 默认 early stopping 指标使用 DBA
训练工作流 MUST 在 `experiment.objective: beam` 或未显式设置 objective 的历史 beam 训练中默认使用验证 DBA/ADBA 作为 early stopping 监控指标。objective-aware 非 beam 训练 MUST 使用对应预测目标的默认主指标：`occlusion` 使用 `val_occlusion_blocked_f1/max`，`position` 使用 `val_position_rmse/min`，`multitask` 使用 `val_multitask_loss/min` 或用户显式配置的可用 multitask 主指标。默认配置 MUST NOT 使用 `top1_val_acc`、`val_acc` 或其它 Top-1 验证准确率别名作为默认 early stopping 指标。

#### Scenario: 默认配置记录 DBA early stopping
- **WHEN** 用户使用未设置 `experiment.objective` 的默认 image、radar、GPS、LiDAR、mmWave 或 fusion 训练配置启动训练
- **THEN** 系统 MUST 将 objective 解析为 `beam`
- **AND** 系统 MUST 在解析后的最终配置中记录 early stopping 监控指标为 `val_adba` 或等价 DBA 别名
- **AND** 系统 MUST 将 early stopping 比较方向记录为越大越好
- **AND** 系统 MUST 不把 `top1_val_acc` 或等价 Top-1 验证准确率别名作为默认 early stopping 指标

#### Scenario: canonical beam 配置默认使用 DBA
- **WHEN** 开发者生成或读取 beam objective canonical 训练配置
- **THEN** canonical 配置 MUST 默认包含 DBA/ADBA early stopping 指标
- **AND** canonical 配置 MUST 不把 Top-1 验证准确率作为默认 early stopping 指标

#### Scenario: objective-aware occlusion 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: occlusion` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_occlusion_blocked_f1`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: max`

#### Scenario: objective-aware position 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: position` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_position_rmse`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`

#### Scenario: objective-aware multitask 默认 early stopping
- **WHEN** 开发者生成或读取 `experiment.objective: multitask` 的训练配置
- **THEN** 解析后的配置 MUST 默认包含 `training.early_stopping_metric: val_multitask_loss`
- **AND** 解析后的配置 MUST 默认包含 `training.early_stopping_mode: min`
- **AND** runtime metadata MUST 记录该 multitask loss 使用的分任务权重

#### Scenario: 显式覆盖 early stopping 指标
- **WHEN** 用户在训练配置或命令行覆盖中显式设置 early stopping 指标为 Top-1、loss 或其它受支持指标
- **THEN** 系统 MUST 使用用户显式指定的指标和比较方向
- **AND** 系统 MUST 校验该指标在当前 objective 的验证结果中真实可用
- **AND** 该覆盖 MUST 不改变项目默认配置继续使用 objective-specific 默认指标的要求

## ADDED Requirements

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
