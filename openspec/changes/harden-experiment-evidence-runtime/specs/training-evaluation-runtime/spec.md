## ADDED Requirements

### Requirement: 训练与评估 runtime 必须保持证据和资源边界
训练、package evaluation 和固定 mask evaluation MUST 对相同 checkpoint 应用一致的 CUDA runtime、profile、GPS 和 normalization 校验。evaluation owner MUST 在退出时关闭其创建的 dataloader workers；默认 evaluation MUST 流式累计指标，除非调用方显式请求 prediction capture。

#### Scenario: 固定 mask 评估 current checkpoint
- **WHEN** fixed-mask evaluator 加载一个 current MMW checkpoint
- **THEN** 它 MUST 校验 checkpoint profile、GPS mode 和 normalization artifact
- **AND** 它 MUST 使用 checkpoint 保存的 train-fitted normalization artifact 而不是重新拟合 scaler

#### Scenario: 默认评估不缓存全量输出
- **WHEN** 常规 validator 或 package evaluator 运行且未请求 capture
- **THEN** runtime MUST 只保留完成 metrics 所需的聚合状态
- **AND** evaluation 完成或失败后 MUST 关闭创建的 worker

### Requirement: development 与 partial 运行必须明确隔离
development 或 partial evaluation MUST 记录其实际 sample/domain/mask 覆盖范围，并不得伪装为完整 comparison evidence。

#### Scenario: 用户限制 batch 或 domain
- **WHEN** evaluator 使用 `max_batches` 或 `max_domains`
- **THEN** 输出 MUST 标记 `development_partial=true` 并记录实际计数
- **AND** 正式 summary MUST 拒绝该输出
