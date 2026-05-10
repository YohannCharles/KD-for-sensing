## MODIFIED Requirements

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
- **AND** 该 helper 的语义 MUST 表示长时序输出对齐，不得作为新 CRAF/MARF `num_pred + 1` fusion head 的兼容承诺

#### Scenario: 输出 slot 不足时报错
- **WHEN** 模型输出 logits 的时间维长度小于 `num_pred`
- **THEN** 训练、验证或评估流程 MUST 报出清晰错误
- **AND** 系统 MUST 不通过重复、padding 或拼接历史 beam 自动补齐 prediction slots

#### Scenario: 诊断预测导出保留 t+1
- **WHEN** viewer prediction export 写出 `confidence_curves` 或 `beam_distribution`
- **THEN** 导出的第一个 horizon MUST 表示 `t+1`
- **AND** 导出逻辑 MUST 不把第一个预测 slot 当作 current beam 丢弃

### Requirement: 训练流程支持 CRAF 输出适配
训练流程 MUST 能消费 CRAF/MARF dict 输出，同时保持现有三元组模型输出兼容。输出适配 MUST 提取 logits、训练 feature、蒸馏 feature 和可选 diagnostics；当 feature-based KD 或 diagnostics 需要真实 feature 时，系统 MUST 使用模型输出的真实 feature 字段，不得用 logits 伪装为 feature 静默继续。

#### Scenario: CRAF dict 输出训练
- **WHEN** 模型 forward 返回包含 `logits`、`input_features` 和 `output_features` 的 dict
- **THEN** 训练流程 MUST 从 dict 中提取 logits 计算任务 loss
- **AND** 训练流程 MUST 使用 dict 中的真实 feature 字段执行需要 feature 的 KD 或 diagnostics
- **AND** 训练流程 MUST 将非核心字段作为 diagnostics 传递给 CRAF/MARF 附加 loss 和日志路径

#### Scenario: dict 输出缺少 logits
- **WHEN** 模型 forward 返回 dict 但不包含受支持的 logits 字段
- **THEN** 输出适配 MUST 报错
- **AND** 训练、验证和诊断导出流程 MUST 不猜测其它 tensor 作为 logits

#### Scenario: 需要 feature 的路径缺少 feature
- **WHEN** 配置启用需要 `input_features` 或 `output_features` 的 KD、auxiliary diagnostics 或 feature 对齐路径
- **AND** 模型输出没有提供对应真实 feature
- **THEN** 训练流程 MUST 报错
- **AND** 系统 MUST 不使用 logits fallback 产生 feature-based loss

#### Scenario: 旧模型三元组输出训练
- **WHEN** 模型 forward 返回 `(pred, input_features, output_features)`
- **THEN** 训练流程 MUST 保持当前 loss、KD 和指标计算语义
- **AND** 三元组中的 feature MUST 被视为真实 feature 输入

#### Scenario: 输出 slot 截取精确对齐
- **WHEN** 模型输出 slot 数已经等于 `num_pred`
- **THEN** 训练流程 MUST 直接使用这些 slot 与未来标签对齐
- **AND** 不得因再次截取而改变语义
