## ADDED Requirements

### Requirement: Sparse pilot CSI observation mask
受限 CSI 输入 MAY 包含 `csi_observation_mask`，用于标记 sparse pilot 观测位置。mask MUST 与 `csi_input` 的 time/subcarrier/antenna 维度对齐，且未观测位置不得携带真实 CSI 值。

#### Scenario: mask 与 csi_input 对齐
- **WHEN** dataset/batch adapter 生成 sparse pilot CSI 输入
- **THEN** `csi_observation_mask` MUST 覆盖 `[T, Nsc, Nant]` 或 batch 后 `[B, T, Nsc, Nant]`
- **AND** `csi_input[..., ~mask, :]` 的 real/imag 值 MUST 为 0 或等价 missing sentinel
- **AND** 完整 clean CSI MUST 只保留在 `csi_target`
