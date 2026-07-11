## REMOVED Requirements

### Requirement: BEV-Fusion 2604 is not a current model
**Reason**: 单独 tombstone spec 不应继续占据 current capability surface。
**Migration**: BEV-Fusion 2604 的 removed-name guard 并入 `retired-route-summary`。

#### Scenario: stale BEV 名称仍被拒绝
- **WHEN** current config 或 registry 收到 BEV-Fusion 2604 名称
- **THEN** 集中 retired-route guard MUST 保持该名称不可构建
