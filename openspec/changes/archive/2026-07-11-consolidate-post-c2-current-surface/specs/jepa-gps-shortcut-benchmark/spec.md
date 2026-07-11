## REMOVED Requirements

### Requirement: JEPA/GPS shortcut benchmark is retired
**Reason**: 独立墓碑折叠到集中 retired-route guard。
**Migration**: 使用 `retired-route-summary` 和参数化旧入口检查；历史细节从 archive/git 查询。

#### Scenario: 集中 guard 承接
- **WHEN** current surface 被检查
- **THEN** shortcut benchmark MUST 继续不可用且不再需要独立 capability spec
