## MODIFIED Requirements

### Requirement: Transformer Encoder 融合与输出契约
CLS-token Transformer fusion MUST 使用包含多头自注意力和前馈网络的 Transformer Encoder 处理 CLS token 与模态 token。模型 MUST 通过 CLS 表示生成未来 beam prediction logits，并兼容现有 `ModelOutput` 适配逻辑。`output_features` 若存在，MUST 用于诊断、auxiliary objective 或 downstream supervised/adaptation workflow，不得作为 KD 兼容要求。

#### Scenario: 输出 future prediction slots
- **WHEN** batch size 为 `B`、配置 `num_pred` 为 `H`、beam 类别数为 `C`
- **THEN** 主 logits MUST 具有形状 `[B, H, C]`
- **AND** 该 logits MUST 能直接传入现有 `select_prediction_slots()`、loss 和 metric 流程

#### Scenario: 输出适配器解析
- **WHEN** CLS-token Transformer fusion forward 返回结果
- **THEN** `adapt_model_output()` MUST 能解析 `logits`、`input_features`、`output_features` 和 diagnostics
- **AND** `output_features` MUST 不被要求服务 RKD 或其它 KD loss

