## ADDED Requirements

### Requirement: 维护索引必须记录架构尺寸基线和统计口径
`docs/maintainer_context_index.yaml` MUST 记录项目架构右尺寸化所需的机器可读基线或定位字段，包括统计来源、统计范围、Python 文件数、function/import 规模、主要子包复杂度、热点 owner、已接受大 owner 和低价值合并候选。统计基线 MUST 明确排除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和其它本地产物。

#### Scenario: 架构审计读取统计基线
- **WHEN** 架构边界测试或维护者审计当前项目结构
- **THEN** 维护索引 MUST 提供足够字段定位统计口径、热点 owner、验证命令和不应纳入源码审计的本地产物路径
- **AND** 审计 MUST NOT 从 generated metadata、ignored runtime artifacts 或历史输出反推当前源码结构

#### Scenario: CodeGraph 或 AST 统计发生漂移
- **WHEN** Python 文件数、function 数、import 数或目录级复杂度发生明显变化
- **THEN** 维护者 MUST 判断变化来自新增 current capability、热点拆分、helper 合并、测试增长还是治理漂移
- **AND** 维护索引或 inventory MUST 更新对应 rationale，而不是只根据数量变化失败

### Requirement: 热点条目必须声明行动、验证和回滚信息
维护索引中的 hotspot entry MUST 为每个登记对象声明 path、symbol 或 owner、priority、status、enforcement、planned action 或 split/consolidation target、public surface policy、rationale、validation commands 和必要 rollback note。状态值 MUST 能表达 `split-next`、`monitor`、`defer-with-rationale`、`right-size-accepted`、`merge-candidate` 和 `facade-budget` 等治理意图。

#### Scenario: 登记 split-next hotspot
- **WHEN** 某个函数、类或 owner 被标记为 `split-next`
- **THEN** 维护索引 MUST 记录拆分目标、headroom 或预算理由、focused tests 和公开行为兼容要求
- **AND** 任务实现 MUST 优先围绕登记的稳定职责边界拆分

#### Scenario: 登记 merge-candidate
- **WHEN** 某个 helper 或 helper 族被标记为 `merge-candidate`
- **THEN** 维护索引 MUST 记录 owner、consolidation targets、不得新增兼容 wrapper 的约束和验证命令
- **AND** 合并完成后索引 MUST 删除旧 helper 作为长期 owner 的暗示

#### Scenario: 登记 right-size-accepted owner
- **WHEN** 某个大 owner 被标记为 `right-size-accepted`
- **THEN** 维护索引 MUST 记录 accepted rationale、保留职责、未来拆分触发条件和 focused tests
- **AND** 架构边界测试 MUST 能区分 accepted owner 与未解释的超预算热点

### Requirement: remediation wave 必须可分阶段实施和回滚
维护索引 MUST 以 remediation wave 或等价结构记录架构整理顺序。每个 wave MUST 声明 target paths、owner module、planned action、public surface policy、validation commands 和 rollback note。Wave MUST 支持 split、consolidate、owner-facade、hard-budget、accepted-size、monitor 和 keep-and-test 等行动类型。

#### Scenario: 开始实施某个 wave
- **WHEN** 开发者准备实施架构整理 wave
- **THEN** 维护索引 MUST 指明该 wave 的目标文件、owner、公开 surface 策略、最小验证命令和回滚边界
- **AND** 开发者 MUST 不把多个无关 wave 混成一次不可定位的大改

#### Scenario: wave 触碰公开 facade
- **WHEN** wave 触碰已登记 CLI、public import owner、benchmark facade 或 baseline reproduction module
- **THEN** 维护索引 MUST 记录该 public surface 是 `keep-public-import`、`thin-owner`、`no-public-surface` 还是 `remove-internal-only`
- **AND** 对应验证命令 MUST 包含架构边界测试和必要 CLI help 或 focused behavior tests

### Requirement: 维护索引必须覆盖新增二级热点
维护索引和 inventory MUST 覆盖 CodeGraph/AST 审计发现的新增二级热点，包括但不限于大型 diagnostics owner、core model owner、config owner、difficulty operator、transform owner 和 runtime cleanup owner。新增二级热点 MAY 标记为 `monitor`、`defer-with-rationale` 或 `keep-and-test`，但 MUST 有明确后续动作或保留理由。

#### Scenario: 新增大型 diagnostics owner
- **WHEN** 审计发现 `jepa_visual_analysis.py`、`run_index.py`、`runtime_artifact_cleanup.py` 或等价 diagnostics owner 体量显著高于普通模块
- **THEN** 维护索引或 inventory MUST 记录其职责边界、拆分候选、暂缓原因和验证命令
- **AND** 若该 owner 向公开 CLI 或 manifest schema 提供输出，拆分计划 MUST 包含行为兼容验证

#### Scenario: 新增 core model 或 config owner 热点
- **WHEN** 审计发现核心模型、config canonical resolver、difficulty operator 或 transform owner 体量较大
- **THEN** 维护索引或 inventory MUST 将其标记为 monitor、keep-and-test 或 split candidate
- **AND** 不得在没有功能变更或测试缺口的情况下为了降低行数强制拆分
