## ADDED Requirements

### Requirement: CLS-token Transformer fusion 辅助头
CLS-token Transformer fusion MUST 支持可选遮挡检测头和位置估算头。辅助头启用时 MUST 从融合后的 CLS 表示或等价 horizon representation 生成输出；辅助头关闭时 MUST 不改变现有模型构建、forward 输入和 beam logits 输出。

#### Scenario: 构建辅助头
- **WHEN** 用户配置 `cls_token_transformer_fusion` 并启用 `auxiliary_heads.occlusion` 和 `auxiliary_heads.position`
- **THEN** 模型 MUST 创建遮挡检测头和位置估算头
- **AND** 模型 MUST 继续创建主 beam prediction head

#### Scenario: 辅助头关闭
- **WHEN** 用户未配置 auxiliary heads 或显式关闭 auxiliary heads
- **THEN** 模型 MUST 保持现有 beam-only output dict 兼容
- **AND** `adapt_model_output()` MUST 继续解析主 logits、input features、output features 和 diagnostics

### Requirement: 辅助输出 horizon 对齐
CLS-token Transformer fusion 的辅助输出 MUST 与主 beam prediction horizon 对齐。遮挡输出 MUST 是每个 future slot 的 logit，位置输出 MUST 是每个 future slot 的二维坐标。

#### Scenario: 五模态辅助输出形状
- **WHEN** 用户构建五模态 CLS-token Transformer fusion，batch size 为 `B`，`num_pred` 为 `H`
- **THEN** forward 返回的主 logits MUST 具有形状 `[B, H, num_classes]`
- **AND** forward 返回的 `occlusion_logits` MUST 具有形状 `[B, H]`
- **AND** forward 返回的 `position` MUST 具有形状 `[B, H, 2]`

#### Scenario: 任意模态子集辅助输出形状
- **WHEN** 用户构建任意合法非空模态子集并启用 auxiliary heads
- **THEN** 模型 MUST 只要求该模态子集的输入张量
- **AND** 辅助输出 shape MUST 只依赖 batch size 和 `num_pred`，不得依赖启用模态数量

### Requirement: 辅助头与模态 mask 兼容
CLS-token Transformer fusion 在使用 `force_modality_mask` 时 MUST 让辅助头基于同一个被 mask 后的 CLS 表示输出，确保 beam、遮挡和位置预测使用一致的有效模态上下文。

#### Scenario: 屏蔽模态后辅助输出仍可用
- **WHEN** forward 传入合法的 `force_modality_mask` 并启用 auxiliary heads
- **THEN** 模型 MUST 使用被 mask 后的 Transformer memory 生成 `occlusion_logits` 和 `position`
- **AND** 输出 diagnostics MUST 继续包含 `effective_modality_mask`

#### Scenario: 空模态 mask 仍被拒绝
- **WHEN** `force_modality_mask` 导致某个样本没有任何可用模态
- **THEN** 模型 MUST 抛出清晰错误
- **AND** 模型 MUST 不生成 beam、遮挡或位置输出

