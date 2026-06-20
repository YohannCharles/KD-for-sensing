## MODIFIED Requirements

### Requirement: OpenSpec capability lifecycle 分类
项目 MUST 维护 OpenSpec capability lifecycle 分类，用于区分当前能力、支撑能力和退役墓碑能力。分类 MUST 至少包含 `current`、`supporting` 和 `retired-tombstone`，并 MUST 覆盖 `openspec/specs/` 下仍作为 current spec 文件保留的每个 capability。已经归档或折叠到集中历史清单的退役路线 MAY 不再拥有 current spec 文件，但 MUST 在 archive、project surface inventory 或 retired route summary 中保留历史边界。

#### Scenario: 每个保留 current spec 被分类
- **WHEN** 开发者或架构边界测试枚举 `openspec/specs/*/spec.md`
- **THEN** 每个仍保留在 current specs 下的 capability MUST 在 lifecycle inventory 中有且仅有一个 lifecycle 分类
- **AND** 未分类 capability MUST 被视为文档生命周期漂移

#### Scenario: lifecycle 分类含义稳定
- **WHEN** 开发者阅读 lifecycle inventory
- **THEN** `current` MUST 表示当前需求契约、可推荐入口或当前运行能力
- **AND** `supporting` MUST 表示被当前 workflow 消费但不作为 standalone 推荐入口的支撑能力
- **AND** `retired-tombstone` MUST 表示只保留退役、拒绝、迁移边界和防回流说明的墓碑能力

#### Scenario: 已归档退役能力不要求 current spec
- **WHEN** 某个退役研究线不再需要当前运行时 guard、配置拒绝边界或专用迁移说明
- **THEN** 项目 MAY 将该 tombstone spec 归档或折叠到集中历史清单
- **AND** 架构健康检查 MUST 不要求 `openspec/specs/` 下继续存在该 retired capability 文件

### Requirement: 退役墓碑 spec 一眼可辨
仍保留在 `openspec/specs/` 下的 `retired-tombstone` capability MUST 在 Purpose 或首个 requirement 中明确说明能力已退役、不属于当前支持面或只作为 migration guard/防回流墓碑保留。退役墓碑 spec MUST NOT 在未加退役限定的上下文中使用当前推荐入口、active mainline、默认 workflow、可运行训练路线等 wording。若墓碑只剩历史说明且无当前 guard 价值，项目 MAY 将其归档或合并到集中历史清单。

#### Scenario: 墓碑 spec 明确退役
- **WHEN** AI agent 或维护者打开保留的 `retired-tombstone` spec
- **THEN** 文档开头 MUST 能直接看出该能力已退役或不属于当前支持面
- **AND** 文件名本身 MUST NOT 被解释为当前可运行入口

#### Scenario: 墓碑 spec 不恢复旧入口
- **WHEN** 保留的墓碑 spec 提到旧 CLI、旧配置、旧模型、旧 dataset 或旧 workflow 名称
- **THEN** 对应段落 MUST 将其描述为退役、拒绝、历史或 migration guard
- **AND** 文档 MUST 不要求新增兼容 alias、root config、console script 或实体 YAML 来恢复旧路线

#### Scenario: 墓碑 spec 可归档
- **WHEN** 退役能力的旧入口已经没有 current docs、config、registry、CLI 或测试迁移价值
- **THEN** 本 change MAY 将该 tombstone spec 移出 current specs
- **AND** archive 或集中历史清单 MUST 继续说明该能力不属于当前支持面
