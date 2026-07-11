## REMOVED Requirements

### Requirement: Config list 和 doctor
**Reason**: 该产品面唯一实现依赖将删除的 surface doctor config scope，并与 inventory/config characterization 重复。
**Migration**: 使用 `docs/project_surface_inventory.md`、config loader/listing helpers、focused characterization 和 static stale-reference checks；不建立替代 doctor/report CLI。

#### Scenario: Config 分类由现有权威承接
- **WHEN** 维护者审计 canonical/generated/local/retired configs
- **THEN** inventory、loader 和 focused tests MUST 提供所需分类事实
- **AND** 项目 MUST 不要求 config doctor command 或 report schema
