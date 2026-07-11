# spec-lifecycle-boundaries Specification

## Purpose
定义 OpenSpec capability lifecycle 分类，避免当前能力、支撑能力和退役墓碑能力在读取、归档或健康检查时被混淆。
## Requirements
### Requirement: OpenSpec capability lifecycle 分类
项目 MUST 维护 OpenSpec capability lifecycle 分类，用于区分当前能力、支撑能力和集中退役边界。`openspec/specs/` 下保留的 capability MUST 分类为 `current` 或 `supporting`；无独立 guard 价值的 retired capability MUST 从 current specs 删除并由 `retired-route-summary`、inventory historical row 或 archive 记录。`retired-tombstone` MAY 只用于确有独立运行时拒绝或外部迁移价值的例外。

#### Scenario: 每个保留 spec 被分类
- **WHEN** 架构边界测试枚举 `openspec/specs/*/spec.md`
- **THEN** 每个保留 capability MUST 在 lifecycle inventory 中有且仅有一个分类
- **AND** 未分类 capability MUST 被视为文档生命周期漂移

#### Scenario: 无独立 guard 的墓碑不占 current spec
- **WHEN** retired capability 只重复“已退役/不得回流”且无独立 config、registry、CLI、docs 或测试 guard
- **THEN** 该 spec MUST 折叠到集中 retired summary 或 archive
- **AND** lifecycle inventory MUST 不再要求该目录存在

#### Scenario: lifecycle 分类含义稳定
- **WHEN** 开发者阅读 lifecycle inventory
- **THEN** `current` MUST 表示当前需求契约或运行能力
- **AND** `supporting` MUST 表示被 current workflow 消费但不作为 standalone 推荐入口的能力
- **AND** 集中 retired summary MUST 表示旧路线只保留历史和拒绝边界

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

### Requirement: supporting capability 不等于 standalone 当前入口
`supporting` capability MAY 保留支撑代码、数据契约、loss、metric、manifest schema、migration guard 或历史读取逻辑，但 MUST 明确不作为 standalone 当前推荐入口。支撑能力被当前 workflow 消费时，文档 MUST 指向实际 current workflow，而不是恢复旧入口。

#### Scenario: TopK 指标保留但 selector/BGAM 支撑退役
- **WHEN** 当前 GPS v2、CSI、benchmark 或通用评估 workflow 复用 Top-K metric、candidate ranking 诊断或 circular metric 支撑代码
- **THEN** lifecycle inventory MAY 将通用指标能力标为 `supporting`
- **AND** 文档 MUST 不把旧 Top8 selector、BGAM-only candidate manifest/loss、训练、plot、compare CLI 或 root config 描述为当前入口

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

### Requirement: 归档后状态收口
OpenSpec change 归档后，维护者 MUST 以 `openspec list --json`、current specs、lifecycle inventory 和 git status 的组合判断项目状态。归档目录存在或未跟踪 archive 文件 MUST 被解释为收口状态，不得单独视为 active change。

#### Scenario: 无 active change 但存在未跟踪 archive
- **WHEN** `openspec list --json` 返回空 change 列表，但 `git status --short` 显示 `openspec/changes/archive/<date>-<change>/` 未跟踪
- **THEN** agent MUST 报告这是归档后待提交或待清理状态
- **AND** agent MUST NOT 将该 archive 目录当作仍在实施的 active change

#### Scenario: current spec 来自归档 change
- **WHEN** 归档 change 新增或修改了 `openspec/specs/<capability>/spec.md`
- **THEN** 同一收口工作 MUST 确认该 capability 的 lifecycle inventory 已更新
- **AND** spec 的 Purpose MUST 使用真实当前能力说明，而不是保留归档 scaffold 占位文本

### Requirement: lifecycle 与文档 caveat 同步
新增 current capability 的 lifecycle 分类 MUST 与 README、主线模型目录、实验协议表和 claim 账本中的状态 caveat 保持一致。若 capability 仍处于 pending、mock/smoke、blocked 或 unverified 状态，文档 MUST 明确说明不可作为正式 claim。

#### Scenario: pending capability 被分类为 current
- **WHEN** lifecycle inventory 将新 capability 标记为 `current`，但该能力的实验结果仍为 `pending`、`mock/smoke`、`blocked` 或 `unverified`
- **THEN** current 文档 MUST 区分“能力入口/契约为 current”和“数值 claim 尚未 verified”
- **AND** 文档 MUST 不把 smoke 或 synthetic 指标写成真实结果

### Requirement: 已完成 active change 必须收口
当 `openspec list --json` 显示 active change 已完成所有任务或 artifact 时，后续表面清理 change MUST 先归档该 change，或在 proposal/design/tasks 中明确说明暂不归档的原因和影响范围。维护者 MUST NOT 将已完成但未归档的 active change 误读为仍在实施的需求。

#### Scenario: 已完成 RBMA change 仍 active
- **WHEN** `add-rbma-prototype-kd-missing-workflow` 或其它 change 显示 status 为 complete
- **THEN** 本次支持面清理 MUST 先执行归档或记录明确 deferral
- **AND** 后续 inventory、current specs 和文档解释 MUST 以归档后的 current surface 或记录的 deferral 为准

#### Scenario: active change 未完成
- **WHEN** active change 仍缺 tasks、design、specs 或实现验证
- **THEN** 维护者 MUST 将其视为当前上下文
- **AND** 新的清理 change MUST 避免覆盖该 active change 范围内的用户工作

### Requirement: 归档脚手架不得进入 current lifecycle
归档 change 生成或修改 current spec 后，维护者 MUST 清理 `TBD` Purpose、模板说明和不完整 lifecycle 分类。current lifecycle 只表示能力契约仍属于当前支持面，不表示 pending/mock/unverified 数值 claim 已经成立。

#### Scenario: 归档后 Purpose 未清理
- **WHEN** current spec 的 Purpose 仍是归档工具生成的占位文本
- **THEN** lifecycle 收口 MUST 被视为未完成
- **AND** 架构边界或 OpenSpec hygiene 检查 MUST 要求补全真实 Purpose

#### Scenario: pending 能力仍为 current
- **WHEN** capability 是 current 入口或契约但结果 claim 仍为 pending、mock/smoke 或 unverified
- **THEN** lifecycle inventory MAY 将其标记为 current
- **AND** mainline catalog、experiment protocols 和 result claims registry MUST 明确标注 claim caveat

### Requirement: 当前架构规格遵循 lifecycle 分类
`project-architecture` spec MUST 与 OpenSpec capability lifecycle inventory 保持一致。已经标记为 `retired-tombstone` 的能力 MUST 只作为退役边界、禁止回流、migration guard 或历史背景出现；标记为 `supporting` 的能力 MUST 不被描述为 standalone 当前推荐入口。

#### Scenario: 退役能力不作为当前热点
- **WHEN** `project-architecture` 提到 HiST/Hist、Raymobtime s008、Top8 selector、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF 或旧 KD
- **THEN** 对应段落 MUST 明确其 retired/supporting 语义
- **AND** 文档 MUST 不要求恢复旧 CLI、旧配置、旧 facade 或旧 root script

#### Scenario: 支撑能力指向当前 workflow
- **WHEN** `project-architecture` 提到仍被当前 workflow 复用的支撑代码
- **THEN** 文档 MUST 指向实际 current workflow
- **AND** 文档 MUST 不把支撑代码所属的旧研究路线描述为当前入口

### Requirement: Lifecycle cleanup cannot weaken retired-route guards
折叠 OpenSpec tombstone、删除历史 wording 或收缩 migration guard 时，项目 MUST 保持 retired route 的实际拒绝边界。若删除某个专属 guard，必须证明普通 unknown-name 错误、集中 retired summary 或其它 guard 仍足以防止旧入口被误判为 current。

#### Scenario: 删除专属 guard 前验证
- **WHEN** 本 change 删除 registry/config/CLI 中某个 retired 名称的专属错误或文档段落
- **THEN** focused tests 或 architecture boundary tests MUST 验证该旧名称仍不会构建、不会被 virtual config 接管、不会出现在 current docs 推荐入口中
- **AND** 删除理由 MUST 说明迁移路径或 unknown-name fallback 是否可接受

### Requirement: Retired tombstones are folded unless they provide active guard value
Retired capability MUST 只在提供独立 registry/config/CLI/docs/tests guard 或外部迁移价值时保留专属 current spec。只重复退役事实的 capability MUST 从 `openspec/specs/` 删除并由集中 summary 覆盖；本轮 27 个 inventory tombstone 中，24 个 MUST 折叠，三个存在 current consumer 的误分类 owner MUST 改为 supporting。

#### Scenario: Tombstone without unique guard
- **WHEN** retired spec 没有独立 guard 或迁移价值
- **THEN** 它 MUST 从 current specs 中移除
- **AND** 集中 retired summary MUST 保留旧名称和非 current 事实

#### Scenario: Retired summary preserves rejection
- **WHEN** 多个 retired specs 被折叠
- **THEN** 项目 MUST 保留代表性旧 CLI/config/module token 与普通 unknown-name 或集中 guard 语义
- **AND** 防回流测试 MUST 继续验证旧入口不会恢复

#### Scenario: Tombstone 分类被 consumer 证据纠正
- **WHEN** source audit 证明 capability 被 current MMW、training startup、U-Mask/AMR/AMBER 或 Scene31-34 workflow 消费
- **THEN** lifecycle inventory MUST 将该 capability 改为 `supporting` 而不是删除
- **AND** supporting spec MUST 只保留真实消费契约，不得借此恢复 standalone CLI 或历史 sweep

### Requirement: 归档后版本控制状态必须成对审计
OpenSpec change 归档后，版本控制状态 MUST 能审计 active change 目录删除与 archive 目录新增是否成对提交。若 `git status --short` 同时显示 `openspec/changes/<change>/` 下文件被删除，并显示 `openspec/changes/archive/<date>-<change>/` 为未跟踪或新增，则实现任务、架构边界测试或最终说明 MUST 要求二者作为同一收口状态处理，或记录明确 deferral 原因。

#### Scenario: 删除 active change 但 archive 未纳入提交
- **WHEN** git status 显示 `D openspec/changes/<change>/...`
- **AND** 同名 dated archive 目录存在于 `openspec/changes/archive/<date>-<change>/`
- **THEN** 归档收口检查 MUST 报告该 change 处于待成对提交状态
- **AND** 维护者 MUST 将 active 目录删除和 archive 新增一起纳入提交，或在最终说明中记录暂缓原因

#### Scenario: archive 目录不代表 active requirement
- **WHEN** `openspec list --json` 不包含某个 change，但 `openspec/changes/archive/<date>-<change>/` 存在或未跟踪
- **THEN** agent 和维护者 MUST 将其解释为归档后版本控制收口状态
- **AND** 不得把该 archive 目录当作仍在实施的 active change 或 current requirement 来源

#### Scenario: 归档成对检查无运行时副作用
- **WHEN** 开发者运行归档完整性检查或架构边界测试
- **THEN** 检查 MUST 只读取 OpenSpec metadata、git-tracked path 状态和文档
- **AND** 检查 MUST 不删除、移动、压缩或重写 active change、archive 目录、`dataset/`、`outputs/`、`logs/`、cache 或 checkpoint
