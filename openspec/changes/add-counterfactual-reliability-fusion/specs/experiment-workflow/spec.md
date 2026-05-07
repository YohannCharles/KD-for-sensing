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

### Requirement: CRAF 稳定化训练工作流
训练流程 MUST 支持 CRAF 稳定化训练配置，包括 warmup gate 固定、CE-only 反事实目标、ignore band、gate/loss schedule 和 softmax gate 诊断。

#### Scenario: warmup 阶段不扰动主任务训练
- **WHEN** CRAF 配置处于 warmup epoch
- **THEN** 训练流程 MUST 执行普通 forward、任务 loss、可配置的 warmup auxiliary loss 和优化步骤
- **AND** 训练流程 MUST 不执行会产生 gate target loss 的 counterfactual supervision

#### Scenario: 反事实启用后写入有效权重
- **WHEN** counterfactual supervision 已启用
- **THEN** `train_log.json` MUST 记录 gate loss 的目标权重和当前有效权重
- **AND** TensorBoard 启用时 MUST 写入等价标量

#### Scenario: 旧训练配置兼容
- **WHEN** CRAF 配置未提供新的稳定化字段
- **THEN** 训练流程 MUST 使用向后兼容默认值
- **AND** 非 CRAF 模型 MUST 不读取或依赖这些字段

### Requirement: CRAF 稳定化实验矩阵
项目 MUST 提供用于定位模态失衡问题的最小 CRAF 消融实验入口。

#### Scenario: token transformer 无 gate baseline
- **WHEN** 用户运行 token transformer 无 gate 配置
- **THEN** 模型 MUST 使用 CRAF tokenizer 与 Transformer backbone
- **AND** 训练流程 MUST 不启用 reliability gate 和 counterfactual gate loss

#### Scenario: CRAF 无反事实 baseline
- **WHEN** 用户运行 CRAF no-counterfactual 配置
- **THEN** 模型 MAY 构建 reliability estimator
- **AND** 训练流程 MUST 固定 gate 或跳过 counterfactual gate supervision

#### Scenario: 固定强模态 prior sanity check
- **WHEN** 用户运行固定 GPS/mmWave 高、image/LiDAR/radar 低的 prior 配置
- **THEN** 训练流程 MUST 使用该 prior 作为诊断 gate 或 dataset prior 输入
- **AND** 该配置 MUST 明确标记为 sanity check 而非默认算法
