# first-class-prediction-tasks Specification

## Purpose
TBD - created by archiving change add-first-class-prediction-tasks. Update Purpose after archive.
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

