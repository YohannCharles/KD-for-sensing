## ADDED Requirements

### Requirement: Scene31-34 final analysis 必须收敛到明确 owner
Scene31-34 final analysis MAY 从多个 per-artifact 脚本收敛到一个 owner command 或 owner module，但 MUST 保持当前 workflow 承诺的表格、图、统计检验、结论文本和展示 artifact 语义。旧脚本删除前 MUST 更新 current docs/spec/tests，使 consolidated owner 成为推荐入口。

#### Scenario: per-artifact exporter 被 owner 覆盖
- **WHEN** profile、significance、paper table、conclusion、CDF、heatmap、sampling 或 presentation artifact 由 consolidated owner 生成
- **THEN** 旧的同职责 per-artifact 脚本 MAY 删除
- **AND** consolidated owner MUST 通过参数、profile 或 view 明确区分输出类型

#### Scenario: final analysis 输出字段保持可对照
- **WHEN** implementation 合并 Scene31-34 final analysis 脚本
- **THEN** 关键 CSV、Markdown、paper table、figure metadata 和 conclusion 输出 MUST 与旧契约做字段级或 snapshot 对照
- **AND** 若字段、排序或筛选条件变化，current spec 和 claim-facing docs MUST 同步更新

### Requirement: Scene31-34 final workflow 不保留历史脚本推荐路径
Scene31-34 current workflow MUST 不推荐已删除的 final analysis 脚本路径。历史说明可以保留，但 MUST 标注 retired 或 historical，并指向 consolidated owner。

#### Scenario: docs 指向 consolidated owner
- **WHEN** README、workflow docs、OpenSpec current specs 或 project inventory 描述 Scene31-34 final analysis
- **THEN** 它们 MUST 指向 consolidated owner command 或 package owner
- **AND** MUST 不把已删除脚本列为当前 required surface
