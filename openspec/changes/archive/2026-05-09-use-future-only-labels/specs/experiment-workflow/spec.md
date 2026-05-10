## ADDED Requirements

### Requirement: 未来标签时隙对齐
训练、验证、评估、诊断预测导出和 KD 相关 loss MUST 使用 `num_pred` 个未来标签时隙。`num_pred=3` 时，系统 MUST 将 label 和预测 slot 解释为 `[t+1, t+2, t+3]`，不得包含当前或历史最后一个 beam。

#### Scenario: 训练 loss 使用未来标签
- **WHEN** 训练流程准备 batch 且 `num_pred: 3`
- **THEN** loss 输入 logits MUST 与 `[t+1, t+2, t+3]` 三个标签时隙对齐
- **AND** flatten 后的 logits 数量 MUST 等于 flatten 后的 labels 数量

#### Scenario: 诊断预测导出保留 t+1
- **WHEN** viewer prediction export 写出 `confidence_curves` 或 `beam_distribution`
- **THEN** 导出的第一个 horizon MUST 表示 `t+1`
- **AND** 导出逻辑 MUST 不把第一个预测 slot 当作 current beam 丢弃

## MODIFIED Requirements

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
- **WHEN** 模型输出 slot 数已经等于 `num_pred`
- **THEN** 训练流程 MUST 能直接使用这些 slot 与未来标签对齐
- **AND** 不得因再次截取而改变语义

#### Scenario: 旧输出多出历史 slot
- **WHEN** 模型输出 slot 数为 `num_pred + 1`
- **THEN** 训练流程 MUST 使用最后 `num_pred` 个 slot 与未来标签对齐
- **AND** 第一个旧 slot MUST 不参与新标签语义下的 loss 或指标计算
