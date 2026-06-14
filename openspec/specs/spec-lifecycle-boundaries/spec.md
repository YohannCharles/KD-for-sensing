# spec-lifecycle-boundaries Specification

## Purpose
定义 OpenSpec capability lifecycle 分类，避免当前能力、支撑能力和退役墓碑能力在读取、归档或健康检查时被混淆。

## Requirements

### Requirement: OpenSpec capability lifecycle 分类
项目 MUST 维护 OpenSpec capability lifecycle 分类，用于区分当前能力、支撑能力和退役墓碑能力。分类 MUST 至少包含 `current`、`supporting` 和 `retired-tombstone`，并 MUST 覆盖 `openspec/specs/` 下每个当前 spec 文件。该分类 MUST 位于 `docs/project_surface_inventory.md` 或等价中心化 inventory 中，不得要求维护完整源码目录清单。

#### Scenario: 每个当前 spec 被分类
- **WHEN** 开发者或架构边界测试枚举 `openspec/specs/*/spec.md`
- **THEN** 每个 spec capability MUST 在 lifecycle inventory 中有且仅有一个 lifecycle 分类
- **AND** 未分类 capability MUST 被视为文档生命周期漂移

#### Scenario: lifecycle 分类含义稳定
- **WHEN** 开发者阅读 lifecycle inventory
- **THEN** `current` MUST 表示当前需求契约、可推荐入口或当前运行能力
- **AND** `supporting` MUST 表示被当前 workflow 消费但不作为 standalone 推荐入口的支撑能力
- **AND** `retired-tombstone` MUST 表示只保留退役、拒绝、迁移边界和防回流说明的墓碑能力

### Requirement: 退役墓碑 spec 一眼可辨
`retired-tombstone` capability MUST 在 Purpose 或首个 requirement 中明确说明能力已退役、不属于当前支持面或只作为 migration guard/防回流墓碑保留。退役墓碑 spec MUST NOT 在未加退役限定的上下文中使用当前推荐入口、active mainline、默认 workflow、可运行训练路线等 wording。

#### Scenario: 墓碑 spec 明确退役
- **WHEN** AI agent 或维护者打开 `retired-tombstone` spec
- **THEN** 文档开头 MUST 能直接看出该能力已退役或不属于当前支持面
- **AND** 文件名本身 MUST NOT 被解释为当前可运行入口

#### Scenario: 墓碑 spec 不恢复旧入口
- **WHEN** 墓碑 spec 提到旧 CLI、旧配置、旧模型、旧 dataset 或旧 workflow 名称
- **THEN** 对应段落 MUST 将其描述为退役、拒绝、历史或 migration guard
- **AND** 文档 MUST 不要求新增兼容 alias、root config、console script 或实体 YAML 来恢复旧路线

### Requirement: supporting capability 不等于 standalone 当前入口
`supporting` capability MAY 保留支撑代码、数据契约、loss、metric、manifest schema、migration guard 或历史读取逻辑，但 MUST 明确不作为 standalone 当前推荐入口。支撑能力被当前 workflow 消费时，文档 MUST 指向实际 current workflow，而不是恢复旧入口。

#### Scenario: TopK 支撑代码保留但 selector 入口退役
- **WHEN** BGAM 或其它当前 workflow 复用 TopK candidate manifest、loss 或 metric 支撑代码
- **THEN** lifecycle inventory MAY 将对应能力标为 `supporting`
- **AND** 文档 MUST 不把旧 Top8 selector 训练、plot、compare CLI 或 root config 描述为当前入口

#### Scenario: 通用 helper 保留但历史 workflow 不复活
- **WHEN** 当前源码保留通用 LOSO、metric、cleanup、migration guard 或 artifact reader helper
- **THEN** lifecycle inventory MUST 区分 helper 的 supporting 地位和旧专用 workflow 的 retired 地位
- **AND** README 或 quickstart MUST 指向当前 workflow，而不是历史 workflow 名称

### Requirement: lifecycle 优先于 spec 文件名和 archive 目录
AI agent 和维护者 MUST 以 active change、当前 specs、lifecycle inventory、README/docs 和源码测试的组合判断当前支持面。Spec 文件名、archive 目录存在、未跟踪 archived change、本地 cache 或 `.pytest_cache` 记录 MUST NOT 单独作为当前支持能力证据。

#### Scenario: 文件名像旧能力但 lifecycle 为墓碑
- **WHEN** `openspec/specs/<old-capability>/spec.md` 的文件名看起来像旧研究能力
- **THEN** lifecycle inventory 和 spec 开头的退役 wording MUST 覆盖文件名暗示
- **AND** agent MUST 将该 spec 解释为防回流或迁移边界，而不是当前入口

#### Scenario: 无 active change 但工作树有归档或缓存噪声
- **WHEN** `openspec list --json` 显示没有 active change，但 git status 中存在未跟踪 archive、ignored `__pycache__`、`.pytest_cache` 或本地运行产物
- **THEN** agent MUST 将这些状态记录为本地收口或缓存噪声风险
- **AND** agent MUST NOT 因这些路径存在而推断当前 specs 已经改变或旧能力重新启用
