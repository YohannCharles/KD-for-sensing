## REMOVED Requirements

### Requirement: JEPA architecture sweep configs stay retired
**Reason**: 独立墓碑折叠到集中 retired-route guard。
**Migration**: Current JEPA pretraining/mean reuse 由其 owner 管理；旧 sweep 从 archive/git 查询。

#### Scenario: 集中 guard 承接
- **WHEN** current configs 被检查
- **THEN** 旧 JEPA architecture sweep MUST 继续不可用且不再需要独立 capability spec
