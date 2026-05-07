## ADDED Requirements

### Requirement: 训练流程支持 CRAF 输出适配
训练流程 MUST 能消费 CRAF dict 输出，同时保持现有三元组模型输出兼容。输出适配 MUST 提取 logits、训练 feature、蒸馏 feature 和可选 diagnostics。

#### Scenario: CRAF dict 输出训练
- **WHEN** 模型 forward 返回包含 `logits` 的 dict
- **THEN** 训练流程 MUST 从 dict 中提取 logits 计算任务 loss
- **AND** 训练流程 MUST 使用 dict 中可用的 feature 字段或安全 fallback 继续运行

#### Scenario: 旧模型三元组输出训练
- **WHEN** 模型 forward 返回 `(pred, input_features, output_features)`
- **THEN** 训练流程 MUST 保持当前 loss、KD 和指标计算语义

#### Scenario: 输出 slot 截取兼容
- **WHEN** 模型输出 slot 数已经等于 `num_pred + 1`
- **THEN** 训练流程 MUST 能直接使用这些 slot 与标签对齐
- **AND** 不得因再次截取而改变语义

### Requirement: 训练流程支持 CRAF 附加 loss
训练流程 MUST 在 CRAF 显式配置时组合普通任务 loss、beam-aware soft label loss、单模态辅助 loss 和 counterfactual gate loss。未启用的 loss 权重 MUST 不影响总 loss。

#### Scenario: 只启用普通任务 loss
- **WHEN** CRAF 附加 loss 权重均为 0
- **THEN** 训练总 loss MUST 等于普通任务 loss 或现有 distiller 组合结果

#### Scenario: 启用 gate loss
- **WHEN** counterfactual gate supervision 产生 gate target
- **THEN** 训练流程 MUST 将 gate loss 按配置权重加入总 loss
- **AND** 日志 MUST 记录 gate loss 摘要

#### Scenario: ignore index 处理一致
- **WHEN** 标签中包含 `-100`
- **THEN** CRAF 附加 loss MUST 跳过这些位置
- **AND** 普通指标计算 MUST 保持现有 ignore index 语义

### Requirement: 评估流程支持 CRAF 输出
评估流程 MUST 能从 CRAF 输出中提取 beam logits 并计算现有 Top-K、DBA 和 loss 指标。评估流程 MUST 不执行训练专用 counterfactual forward。

#### Scenario: CRAF 模型评估
- **WHEN** 用户评估 CRAF checkpoint
- **THEN** 评估流程 MUST 提取 CRAF logits
- **AND** 评估流程 MUST 保存与现有模型一致的 metrics 文件

#### Scenario: 评估跳过 counterfactual
- **WHEN** 配置中 counterfactual training 曾启用
- **THEN** 评估流程 MUST 不执行 drop-forward gate supervision
- **AND** 评估结果 MUST 只反映正常 effective modality mask 下的预测表现

### Requirement: CRAF 日志与运行产物
训练输出 MUST 在 CRAF diagnostics 可用时保存 reliability、counterfactual 和 auxiliary loss 摘要，并 MUST 保持现有 `train_log.json`、`metrics.json` 和 TensorBoard 输出兼容。

#### Scenario: train_log 记录 CRAF 字段
- **WHEN** CRAF 训练完成至少一个 epoch
- **THEN** `train_log.json` MUST 包含 CRAF 附加 loss 和每模态 reliability 的 epoch 摘要

#### Scenario: final_config 保存 CRAF 配置
- **WHEN** CRAF 训练启动
- **THEN** `final_config.yaml` MUST 保存实际生效的 CRAF 模型、loss、counterfactual 和 modality dropout 配置

#### Scenario: 旧模型日志结构兼容
- **WHEN** 用户训练非 CRAF 模型
- **THEN** 输出日志 MUST 保持现有字段兼容
- **AND** CRAF 专属字段 MAY 缺省

### Requirement: CRAF smoke test 工作流
项目 MUST 提供可在 conda 环境中运行的 CRAF smoke test，覆盖模型构建、forward、loss、backward、验证和日志写入的核心路径。

#### Scenario: synthetic CRAF 短训练
- **WHEN** 开发者运行 CRAF synthetic 或小数据短训练测试
- **THEN** 训练流程 MUST 完成至少一个 optimizer step
- **AND** 验证流程 MUST 产出 metrics

#### Scenario: CRAF 配置加载测试
- **WHEN** 开发者运行配置加载测试
- **THEN** CRAF 示例配置和 baseline 示例配置 MUST 能通过 config loader 解析
