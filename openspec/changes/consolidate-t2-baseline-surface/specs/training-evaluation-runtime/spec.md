## ADDED Requirements

### Requirement: T2 runtime 仅保留 same-model consistency
训练 runtime MUST 仅保留 T2 所需的 embedded full-modal teacher CE 和 same-model temporal superset consistency。该 extension MUST 在训练期执行，evaluation MUST 不执行第二次 teacher forward；T2 disabled path 与 S1 MUST 不产生额外 forward 或外部 teacher artifact 读取。

#### Scenario: S1 不执行 superset consistency
- **WHEN** S1 recipe 将 superset consistency 关闭
- **THEN** trainer MUST 不保存 superset payload 或执行第二次 model forward
- **AND** 训练仍能计算 T2 共用的 beam、teacher CE、BPA 和 router loss

## REMOVED Requirements

### Requirement: weak-pattern KD
**Reason**: 该未启用的 KD 分支不属于 T2/baseline 闭包。
**Migration**: 不提供替代；T2 仅使用 same-model superset consistency。

### Requirement: lightweight latent prediction probe
**Reason**: 该可选 probe 不被 T2/baseline recipe 或 active T2 change 使用。
**Migration**: 不提供替代。
