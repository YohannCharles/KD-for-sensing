## ADDED Requirements

### Requirement: CSI reconstruction supervision for physics baseline
系统 MUST 允许 physics-informed baseline 将当前完整 clean CSI 作为 `csi_target` reconstruction target。该 target MUST 沿用 `[T, Nsc, Nant, 2]` real/imag 末维契约，MUST 不作为默认 sensing input，并且在目标缺失时 MUST 通过 mask 跳过 CSI reconstruction loss。

#### Scenario: clean CSI 作为 reconstruction target
- **WHEN** 配置启用 physics supervision 和 `physics.loss.csi_reconstruction.enabled=true`
- **THEN** dataset/batch adapter MUST 提供 clean `csi_target` 或显式 unavailable mask
- **AND** reconstruction loss MUST 使用 clean CSI target 计算 NMSE/MSE
- **AND** metadata MUST 记录 CSI target 来源和是否使用 degradation

#### Scenario: 受限 CSI 才能作为模型输入
- **WHEN** 配置启用 `data.use_csi_input=true`
- **THEN** dataset/batch adapter MUST 根据 `data.csi_input_mode` 提供 `csi_input`
- **AND** `history`、`partial`、`noisy`、`compressed` 模式 MUST 不直接暴露当前完整 CSI
- **AND** 只有 `oracle_full` 模式在显式 `allow_oracle_full_csi_input=true` 时 MAY 将当前完整 CSI 作为模型输入

#### Scenario: CSI shape 对齐失败可诊断
- **WHEN** `h_hat` 和 clean CSI target 的 subcarrier、antenna 或 time/horizon 维度无法按配置对齐
- **THEN** physics loss MUST 抛出包含 `h_hat` shape、CSI shape、num_subcarriers 和 antenna 维度的错误
- **AND** 系统 MUST 不静默 broadcast 或截断到错误维度

#### Scenario: 未启用 CSI 不要求 CSI 列
- **WHEN** physics-informed 配置关闭 CSI 输入和 CSI reconstruction loss
- **THEN** dataset MUST 不要求 `csi*` 列
- **AND** loss diagnostics MUST 标记 CSI reconstruction disabled 而不是 unavailable error
