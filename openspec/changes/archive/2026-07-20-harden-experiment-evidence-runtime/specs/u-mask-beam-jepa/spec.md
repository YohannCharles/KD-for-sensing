## ADDED Requirements

### Requirement: U-Mask active branch 必须匹配训练与参数语义
U-Mask MUST 将 inactive classifier/prototype/router branch 冻结并从 optimizer trainable parameter 集合排除；metadata MUST 声明 active head 与 trainable parameter count。`reliability_mean` 在 router oracle weight 为零时 MUST 不构造 oracle-loss gradient graph，但仍可输出明确的 disabled diagnostics。

#### Scenario: T2-CLS 构建
- **WHEN** T2-CLS 使用 classifier head 构建模型
- **THEN** prototype branch MUST 不参与 optimizer 或 trainable parameter count
- **AND** metadata MUST 记录 classifier 为 active head

#### Scenario: reliability mean 训练
- **WHEN** `fusion_type=reliability_mean` 且 router oracle weight 为零
- **THEN** training MUST 不计算 router oracle loss
- **AND** diagnostics MUST 标记该项为 disabled
