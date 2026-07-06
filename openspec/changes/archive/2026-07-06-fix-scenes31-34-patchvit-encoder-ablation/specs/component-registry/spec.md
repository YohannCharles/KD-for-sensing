## ADDED Requirements

### Requirement: Lightweight PatchViT frame encoder registry
默认组件导入 MUST 注册轻量 PatchViT 帧级 encoder，使其可通过 `ENCODERS` 在 `modular_sequence` 中选择。该 encoder MUST 复用已有 patch visual token encoder 的卷积 patch embedding 与 Transformer 层，并输出 `[B, T, D]` 帧级特征。

#### Scenario: 构建轻量 PatchViT encoder
- **WHEN** 构建流程调用默认组件导入后查询 `ENCODERS`
- **THEN** registry MUST 包含 `lightweight_patchvit_frame`
- **AND** MAY 包含等价别名 `patchvit_frame`

#### Scenario: PatchViT encoder forward 契约
- **WHEN** `lightweight_patchvit_frame` 收到 `[B, T, C, H, W]` 输入
- **THEN** encoder MUST 输出 `[B, T, output_dim]`
- **AND** metadata MUST 记录其 visual token encoder 类型、token source、token count、pooling 和 input channel count
