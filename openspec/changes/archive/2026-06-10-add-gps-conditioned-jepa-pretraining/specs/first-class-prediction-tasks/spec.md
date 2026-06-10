## ADDED Requirements

### Requirement: GPS-conditioned JEPA 自监督 objective
系统 MUST 支持 `experiment.objective: gps_conditioned_jepa`，用于 GPS 条件化 JEPA latent prediction 预训练。该 objective MUST 使用 image 和 GPS 输入构造自监督 latent target，MUST 不要求 beam label、occlusion label、position target、LOS label 或 link quality target，并 MUST 不计算 beam Top-K、DBA、occlusion、position、LOS 或 link 指标。

#### Scenario: 解析 JEPA objective
- **WHEN** 用户加载 `experiment.objective: gps_conditioned_jepa` 的配置
- **THEN** objective metadata MUST 将主 loss 解析为 `jepa`
- **AND** 默认 early stopping metric MUST 为 `val_jepa_loss`
- **AND** 默认 early stopping mode MUST 为 `min`

#### Scenario: JEPA objective 不要求 beam target
- **WHEN** JEPA 训练 batch 包含 image 和 GPS 但不包含 `target_beam`
- **THEN** 训练流程 MUST 能准备自监督 JEPA target
- **AND** 系统 MUST 不调用 beam label 准备、beam CE/Focal loss 或 beam Top-K 指标

#### Scenario: JEPA objective 要求 image 和 GPS
- **WHEN** JEPA objective 的 batch 缺少 image 或 GPS 字段
- **THEN** 系统 MUST 拒绝训练或验证该 batch
- **AND** 错误信息 MUST 包含缺失字段和 `gps_conditioned_jepa` objective 名称

### Requirement: JEPA objective 指标和日志契约
系统 MUST 为 `gps_conditioned_jepa` objective 提供独立的 available metrics、history fields、TensorBoard scalar 映射和 runtime metadata。该 objective 的公开验证指标 MUST 至少包含 `val_loss` 和 `val_jepa_loss`，且 `val_jepa_loss` MUST 是默认主指标。

#### Scenario: JEPA validation metrics
- **WHEN** 验证或评估 `gps_conditioned_jepa` objective
- **THEN** validation metrics MUST 包含 `val_jepa_loss` 和 `val_loss`
- **AND** `available_metrics` MUST 只暴露 JEPA objective 可用指标和通用 loss 指标
- **AND** validation metrics MUST NOT 暴露 `val_adba`、`val_acc`、`val_beam_top1`、`val_occlusion_blocked_f1`、`val_position_rmse`、`val_los_f1` 或 `val_link_mae`

#### Scenario: JEPA TensorBoard scalar 隔离
- **WHEN** 当前 objective 为 `gps_conditioned_jepa`
- **THEN** TensorBoard scalar MUST 包含 JEPA train/validation loss、mask ratio 和 EMA decay 中已产生的正式字段
- **AND** TensorBoard scalar MUST NOT 包含 beam Top-K、DBA、occlusion、position、LOS 或 link quality tag

#### Scenario: JEPA runtime metadata
- **WHEN** JEPA 训练创建或完成运行产物
- **THEN** `final_config.yaml` 或运行 metadata MUST 记录 `objective: gps_conditioned_jepa`
- **AND** metadata MUST 记录主 loss、主 metric、metric mode、启用 targets、启用 model outputs 和 pretraining kind

### Requirement: JEPA objective early stopping 校验
系统 MUST 根据 `gps_conditioned_jepa` objective 的 available metrics 校验 early stopping metric。用户显式配置 beam、occlusion、position、LOS 或 link metric 作为 JEPA early stopping metric 时，系统 MUST 拒绝继续训练并报告 JEPA objective 可用指标。

#### Scenario: 拒绝 beam early stopping metric
- **WHEN** 配置设置 `experiment.objective: gps_conditioned_jepa` 且 `training.early_stopping_metric: val_adba`
- **THEN** early stopping metric 校验 MUST 失败
- **AND** 错误信息 MUST 包含 `gps_conditioned_jepa` 和可用指标列表

#### Scenario: 接受 JEPA early stopping metric
- **WHEN** 配置设置 `experiment.objective: gps_conditioned_jepa` 且 `training.early_stopping_metric: val_jepa_loss`
- **THEN** early stopping metric 校验 MUST 通过
- **AND** checkpoint metadata MUST 记录 primary metric 为 `val_jepa_loss` 且 mode 为 `min`
