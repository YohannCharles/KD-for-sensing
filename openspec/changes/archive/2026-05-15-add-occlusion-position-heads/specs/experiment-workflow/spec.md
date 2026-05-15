## ADDED Requirements

### Requirement: 训练流程支持多任务辅助 loss
训练流程 MUST 在保持现有 beam/KD 基础 loss 的前提下支持可选多任务辅助 loss。辅助 loss MUST 只在配置启用且 batch/model 均提供对应字段时计算；否则训练流程 MUST 保持现有 beam-only 行为。

#### Scenario: no-KD 多任务训练
- **WHEN** 用户运行启用遮挡和位置辅助任务的 no-KD fusion 训练
- **THEN** 训练流程 MUST 计算 beam CE、遮挡 BCE 和位置 MSE
- **AND** optimizer step MUST 使用三者加权后的总 loss
- **AND** train log MUST 记录每个 loss 分量

#### Scenario: KD 多任务训练
- **WHEN** 用户运行启用辅助任务的 logits KD 或 RKD fusion 训练
- **THEN** 训练流程 MUST 保留既有 KD 基础 loss 计算
- **AND** 训练流程 MUST 将辅助 loss 加到 student 总 loss
- **AND** teacher 模型 MUST 不被要求输出辅助头，除非配置显式启用 teacher auxiliary supervision

#### Scenario: 辅助字段缺失
- **WHEN** 配置启用辅助 loss 但 batch 或模型输出缺少必要字段
- **THEN** 训练流程 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出缺失的是 dataset target 还是 model auxiliary output

### Requirement: 验证和评估输出辅助指标
验证和评估流程 MUST 在启用多任务辅助监督时输出遮挡和位置指标，同时保留现有 Top-K、DBA、loss、degradation baseline 和 modality subset 评估语义。

#### Scenario: 验证输出遮挡指标
- **WHEN** 验证流程收到 `occlusion_logits` 和 `occlusion_label`
- **THEN** validation metrics MUST 包含遮挡 accuracy 和 blocked-class F1
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: 验证输出位置指标
- **WHEN** 验证流程收到 `position` 和 `position_target`
- **THEN** validation metrics MUST 包含 position RMSE
- **AND** epoch log 和 TensorBoard MUST 记录对应标量

#### Scenario: beam 指标保留
- **WHEN** 多任务辅助监督启用
- **THEN** 验证和评估流程 MUST 继续输出 beam Top-K、DBA、ATop-3、ATop-5 和 ADBA
- **AND** early stopping 默认 MUST 继续支持现有 `val_adba` 配置

### Requirement: 多任务运行产物可复现
训练和评估流程 MUST 在运行产物中记录多任务配置、遮挡阈值、辅助目标统计、loss 权重和辅助指标，确保后续评估和复现实验能加载相同的标签生成状态。

#### Scenario: final config 记录多任务状态
- **WHEN** 训练启用多任务辅助监督
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录遮挡阈值、阈值分位数、位置目标来源和 loss 权重
- **AND** checkpoint 或 normalization artifacts MUST 记录独立评估所需的辅助目标统计

#### Scenario: train log 记录辅助指标历史
- **WHEN** 训练至少完成一个 epoch 且启用多任务辅助监督
- **THEN** `train_log.json` MUST 包含遮挡和位置指标历史
- **AND** `training_outputs.npz` MUST 保存可画曲线的辅助 loss 或指标数组

