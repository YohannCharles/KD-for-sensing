## ADDED Requirements

### Requirement: 导航文档按 spec lifecycle 判断当前支持面
AI 维护导航文档 SHALL 指导维护者在读取 `openspec/specs/` 时先识别 capability lifecycle，再判断需求内容。导航文档 MUST 说明 `current`、`supporting` 和 `retired-tombstone` 的读取语义，并 MUST 指向 lifecycle inventory 或等价中心化分类来源。

#### Scenario: 读取 current specs 前先看 lifecycle
- **WHEN** AI agent 需要判断某个 OpenSpec capability 是否属于当前支持面
- **THEN** 导航文档 MUST 要求先查看 lifecycle 分类
- **AND** `retired-tombstone` spec MUST 被解释为退役边界、防回流或 migration guard，而不是当前运行入口

#### Scenario: supporting 能力不被误判为推荐入口
- **WHEN** lifecycle 分类为 `supporting`
- **THEN** 导航文档 MUST 说明该能力只能作为当前 workflow 的支撑能力理解
- **AND** agent MUST 继续查 README、inventory 或 current workflow spec 来确认实际推荐入口

### Requirement: 导航文档覆盖归档未收口和本地缓存噪声
AI 维护导航文档 SHALL 明确区分 active change、archived change、未跟踪归档目录、ignored runtime/cache artifacts 和 `.pytest_cache`。导航文档 MUST 说明这些状态不能单独覆盖当前 specs 或 README/docs 推荐入口。

#### Scenario: archived change 目录存在但不是 active change
- **WHEN** `openspec list --json` 不列出某个 change，但 `openspec/changes/archive/` 或 git status 中存在相关目录
- **THEN** 导航文档 MUST 要求将其视为历史记录或版本控制收口问题
- **AND** agent MUST 不把 archived change 当作正在实施的 active change

#### Scenario: pytest cache 不作为当前测试红点
- **WHEN** `.pytest_cache/v/cache/lastfailed` 或 ignored `__pycache__` 提示旧测试或本地缓存状态
- **THEN** 导航文档 MUST 说明这些是 ignored runtime artifacts
- **AND** agent MUST 通过实际 pytest 命令或当前测试文件判断真实失败状态

### Requirement: 导航文档提示语义冲突处理方式
AI 维护导航文档 SHALL 说明当同一当前 spec 内部存在旧 active wording 与退役要求冲突时，维护者 MUST 优先创建或执行 OpenSpec 清理 change，而不是让 agent 自行选择一段文字作为事实。导航文档 MUST 鼓励将冲突收敛到 current/supporting/retired lifecycle 分类和当前 README/inventory 对齐。

#### Scenario: 当前 spec 内部出现冲突 wording
- **WHEN** `project-architecture` 或其它 current spec 同时把某路线描述为 active mainline 和 retired
- **THEN** 导航文档 MUST 要求把它视为规格漂移
- **AND** 后续变更 MUST 清理旧 active wording 或明确 supporting/retired 分类
