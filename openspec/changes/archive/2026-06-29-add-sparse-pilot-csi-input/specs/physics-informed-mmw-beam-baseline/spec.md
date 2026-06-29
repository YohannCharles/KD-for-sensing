## ADDED Requirements

### Requirement: Sparse pilot CSI input mode
系统 MUST 为 physics-informed MMW baseline 提供 `csi_input_mode=sparse_pilot`，将当前 clean CSI target 转换为带观测 mask 的 sparse pilot observation。该模式 MUST 不把未观测 CSI 值传给模型，MUST 保留完整 `csi_target` 仅用于 reconstruction supervision、beam gain 诊断或 oracle upper-bound 对照。

#### Scenario: structured sparse pilot observation
- **WHEN** 配置设置 `use_csi_input=true`、`csi_input_mode=sparse_pilot`、`pilot_pattern=grid`
- **THEN** adapter MUST 返回 `csi_input`，其 shape 与 `csi_target` 相同
- **AND** 未观测 subcarrier/antenna 位置 MUST 为 0
- **AND** adapter MUST 返回 `csi_observation_mask`，标记观测到的 pilot 位置
- **AND** metadata MUST 记录 pattern、subcarrier stride、antenna stride 和 observed fraction

#### Scenario: sparse pilot 不替代完整监督
- **WHEN** sparse pilot 输入启用
- **THEN** 模型 forward MUST 只消费 `csi_input`
- **AND** CSI reconstruction loss MUST 继续使用完整 clean `csi_target`
- **AND** run metadata MUST 将该输入标记为受限 `csi_observed`，而不是 oracle full CSI

#### Scenario: partial CSI 只作为 ablation
- **WHEN** 文档或配置描述 `partial` CSI 输入
- **THEN** 系统 SHOULD 将其标记为 debug/ablation proxy
- **AND** sparse pilot SHOULD 作为 physics-informed MMW baseline 的推荐受限 CSI 输入配置
