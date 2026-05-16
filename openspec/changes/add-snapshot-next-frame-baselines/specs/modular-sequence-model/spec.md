## ADDED Requirements

### Requirement: Snapshot frame representation core
模块化序列模型 MUST 提供 `snapshot_frame` representation core，用于无历史窗口的当前帧预测。该 core MUST 支持单模态 `[B, 1, D]` 输入和多模态 `[B, K, 1, D]` 输入，并输出可被现有 heads 消费的 `[B, 1, D_out]` 表示。

#### Scenario: 单模态 snapshot core
- **WHEN** `modular_sequence` 配置启用单个模态并设置 `representation_core.type: snapshot_frame`
- **THEN** core MUST 接收该模态 projector 输出 `[B, 1, d_model]`
- **AND** core MUST 输出 `[B, 1, output_dim]`
- **AND** beam head MUST 生成 `[B, 1, num_classes]` logits

#### Scenario: 多模态 snapshot core
- **WHEN** `modular_sequence` 配置启用多个模态并设置 `representation_core.type: snapshot_frame`
- **THEN** core MUST 接收堆叠后的 `[B, K, 1, d_model]` 模态表示
- **AND** core MUST 只在当前帧的 `K` 个模态表示之间执行融合
- **AND** core MUST 输出 `[B, 1, output_dim]`

#### Scenario: 拒绝历史时间维
- **WHEN** `snapshot_frame` core 收到时间维 `T` 不等于 1 的输入
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 指出 snapshot baseline 需要 `seq_len=1` 和 `num_pred=1`

#### Scenario: 不创建 GRU 模块
- **WHEN** 开发者构建启用 `snapshot_frame` core 的 `modular_sequence` 模型
- **THEN** 模型 MUST 不创建 `single_gru` 或 `early_concat_gru`
- **AND** 模型模块树中 MUST 不包含 GRU、RNN 或 LSTM 子模块

### Requirement: Snapshot core 辅助 head 兼容
`snapshot_frame` core MUST 保持模块化模型的 head 输出契约。启用遮挡或位置 auxiliary heads 时，辅助输出 MUST 与 `num_pred=1` 的 next-frame horizon 对齐。

#### Scenario: Snapshot 遮挡输出
- **WHEN** snapshot `modular_sequence` 配置启用 `auxiliary_heads.occlusion`
- **THEN** forward 输出 MUST 包含形状 `[B, 1]` 的 `occlusion_logits`
- **AND** 主 beam logits MUST 继续保持 `[B, 1, num_classes]`

#### Scenario: Snapshot 位置输出
- **WHEN** snapshot `modular_sequence` 配置启用 `auxiliary_heads.position`
- **THEN** forward 输出 MUST 包含形状 `[B, 1, 2]` 的 `position`
- **AND** 输出 MUST 能被现有 objective-aware loss 和 metrics 消费
