## ADDED Requirements

### Requirement: current spec 内部语义一致
`current` lifecycle 的 OpenSpec capability MUST 在同一 spec 内保持当前支持面语义一致。若同一 spec 同时包含 active mainline、retired/supporting、migration guard 或 historical wording，文档 MUST 明确区分其适用范围；不得让旧 active workflow 与当前拒绝边界并存而无解释。

#### Scenario: current spec 不保留旧 active workflow
- **WHEN** current spec 中保留了旧 KD、teacher/student、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual、camera residual 或其它退役路线描述
- **THEN** 对应段落 MUST 标记为 retired、historical、supporting 或 migration guard
- **AND** 文档 MUST 不同时要求实现该旧路线作为当前 active workflow

#### Scenario: current spec 发生语义冲突
- **WHEN** 一个 current spec 内部既要求某路线作为当前默认入口，又在其它段落要求该路线退役或拒绝
- **THEN** 该状态 MUST 被视为规格漂移
- **AND** 维护者 MUST 通过 OpenSpec change 将其收敛为单一 current、supporting 或 retired 叙事

### Requirement: lifecycle 决策优先级
维护者和 AI agent MUST 结合 active change、current specs、lifecycle inventory、README/docs 和源码测试判断当前支持面。若 current spec 与 lifecycle inventory 或 README 当前入口冲突，MUST 优先将其视为待清理的规格漂移，而不是任选一段作为事实。

#### Scenario: current spec 与 inventory 冲突
- **WHEN** lifecycle inventory 将某能力标记为 retired-tombstone 或 supporting，但 current spec 中存在未加限定的 active mainline wording
- **THEN** agent MUST 将其报告为 lifecycle/wording 漂移
- **AND** agent MUST NOT 根据该 active wording 恢复旧 CLI、配置、registry 名称或训练入口

#### Scenario: historical report 与 current docs 冲突
- **WHEN** 历史报告、运行流水账或 archive 中的命令与 README、experiment matrix、mainline catalog 或 current spec 的当前口径冲突
- **THEN** historical report MUST 只作为演进背景
- **AND** 当前推荐入口 MUST 以 current docs、lifecycle inventory 和 current specs 的收敛结果为准

### Requirement: 新文档能力 lifecycle 分类
新增文档治理类 capability MUST 在 lifecycle inventory 中明确分类，并说明其与 README、experiment matrix、project surface inventory、baseline report 和 OpenSpec specs 的职责边界。

#### Scenario: mainline documentation capability 被分类
- **WHEN** `mainline-experiment-documentation` spec 被新增
- **THEN** lifecycle inventory MUST 将其分类为 `current`
- **AND** inventory MUST 说明该能力维护当前主线模型目录、实验协议表和结果账本，不替代 OpenSpec 行为契约

#### Scenario: 文档职责边界清晰
- **WHEN** 维护者阅读文档生命周期分类
- **THEN** README MUST 被描述为 quickstart 和索引
- **AND** 主线模型目录、实验协议表、结果账本、baseline current summary 和历史流水账 MUST 有各自职责边界
