## ADDED Requirements

### Requirement: 共享 evaluation pass
训练验证、force-mask subset 验证和 standalone evaluate MUST 复用同一个 evaluation pass 完成 batch 准备、model forward、objective loss、输出收集、指标聚合和 available metrics 生成。各入口 MAY 对结果做输出包装或文件写出，但 MUST 不复制核心 forward/loss/collect 逻辑。

#### Scenario: 普通验证使用共享 pass
- **WHEN** 训练流程在 epoch 结束后调用 validation
- **THEN** validation MUST 通过共享 evaluation pass 计算 loss、Top-K、DBA 和 objective 指标
- **AND** 返回的公开 metrics 键 MUST 保持与变更前兼容

#### Scenario: force-mask subset 使用共享 pass
- **WHEN** evaluation 配置启用 modality subset 或 force mask 验证
- **THEN** subset validation MUST 使用同一个 evaluation pass 并传入 mask 选项
- **AND** subset 结果 MUST 包含与普通验证一致的 objective metadata 和 available metrics

#### Scenario: standalone evaluate 使用共享 pass
- **WHEN** 用户通过评估入口运行 checkpoint evaluate
- **THEN** evaluate MUST 使用共享 evaluation pass 计算指标
- **AND** 保存的报告 MUST 与训练验证使用同一套 objective 指标语义

### Requirement: 评估指标写出与 runtime metadata 对齐
训练和评估写出的 metrics/report MUST 包含 objective runtime metadata、primary metric、available metrics 和已启用模态信息。该 metadata MUST 来自 objective 与 modality resolution 层，而不是入口各自手写推导。

#### Scenario: 评估报告记录 objective metadata
- **WHEN** 用户评估 `experiment.objective: occlusion` 的模型
- **THEN** 评估报告 MUST 记录 objective 名称、primary loss、primary metric、metric mode、enabled targets 和 enabled heads
- **AND** 这些字段 MUST 与训练 final config 中的 prediction objective metadata 一致

#### Scenario: 评估报告记录启用模态
- **WHEN** 用户评估 GPS+mmWave fusion 模型
- **THEN** 评估报告 MUST 记录启用模态为 `["gps", "mmwave"]`
- **AND** 该模态集合 MUST 由统一模态解析逻辑产生
