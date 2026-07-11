## REMOVED Requirements

### Requirement: Visual diagnostics do not restore deleted CLIs
**Reason**: 独立墓碑折叠到集中 retired-route guard。
**Migration**: Current model/evaluation owner 只保留实际 diagnostics；旧 CLI 从 archive/git 查询。

#### Scenario: 集中 guard 承接
- **WHEN** current CLI 被检查
- **THEN** 旧 visual diagnostics MUST 继续不可用且不再需要独立 capability spec
