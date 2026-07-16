## MODIFIED Requirements

### Requirement: T2 runtime 仅保留 same-model consistency

训练 runtime MUST 仅保留 T2 所需的 embedded full-modal teacher CE 和 same-model temporal superset consistency。evaluation MUST 不执行第二次 teacher forward；T2 disabled path 与 S1 MUST 不产生额外 forward 或外部 artifact 读取。

#### Scenario: S1 不执行 superset consistency

- **WHEN** S1 recipe 将 superset consistency 关闭
- **THEN** trainer MUST 不保存 superset payload 或执行第二次 model forward
- **AND** 训练仍能计算 T2 共用的 beam、teacher CE、BPA 和 router loss
