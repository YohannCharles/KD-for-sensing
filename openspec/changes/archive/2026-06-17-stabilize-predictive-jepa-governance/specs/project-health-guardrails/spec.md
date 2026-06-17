## ADDED Requirements

### Requirement: 归档后 current spec 治理检查
项目健康护栏 MUST 检查归档后进入 `openspec/specs/` 的 current capability 同时具备 lifecycle inventory 分类、非占位 Purpose 和对应文档 caveat。检查 MUST 不读取真实 `dataset/`、`outputs/`、checkpoint、cache 或 logs。

#### Scenario: 新 current spec 缺少 lifecycle 分类
- **WHEN** `openspec/specs/<capability>/spec.md` 存在且该 capability 不在 `docs/project_surface_inventory.md` 的 lifecycle inventory 中
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向补充 `current`、`supporting` 或 `retired-tombstone` 分类

#### Scenario: 归档生成的 Purpose 未清理
- **WHEN** current spec 的 `## Purpose` 为空、长度不足或包含 `TBD`
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应 spec 文件

### Requirement: current JEPA 合法语境不被旧路线 guard 误判
项目健康护栏 MUST 继续拒绝退役路线 active wording，但 MUST 允许 current JEPA specs 和 diagnostics 中对现有 GPS-query baseline compatibility、condition-id 禁用字段和 forbidden-field diagnostics 的合法描述。

#### Scenario: GPS-query compatibility wording 被允许
- **WHEN** current JEPA spec 描述 `GPS-query` 或 `gps_query_pool` 作为现有 baseline compatibility、对照模型或默认行为兼容性
- **THEN** retired-route wording guard MUST 不把该行判定为退役路线回流
- **AND** 文档 MUST 不把该 baseline 写成旧 KD、HiST、Top8 selector standalone、GPS residual 或 camera residual 路线

#### Scenario: forbidden condition 字段诊断被允许
- **WHEN** current source 或 spec 记录 `condition_id_consumed=false`、`blocked_condition_fields`、`forbidden_condition_fields`、`gps_condition` 或 `image_condition`
- **THEN** 健康护栏 MUST 将其解释为防止 condition-aware router 的诊断或安全边界
- **AND** 只有在同一上下文把这些字段描述为模型直接输入或当前 router 入口时才应失败
