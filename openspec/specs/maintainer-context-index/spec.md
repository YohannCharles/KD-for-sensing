# maintainer-context-index Specification

## Purpose
定义中心化、机器可读的维护上下文索引，使 AI agent、维护者和架构边界测试能从稳定位置读取项目任务路由、治理事实、生命周期入口和无运行副作用边界，同时不替代 OpenSpec requirements、README quickstart、AGENTS 操作规则或项目表面积 inventory 的审计解释职责。
## Requirements
### Requirement: 中心化维护上下文索引
项目 SHALL 提供一份稳定路径的中心化维护上下文索引，用于让 AI agent、维护者和架构边界测试读取项目治理事实。该索引 MUST 使用机器可读格式，MUST 位于 `docs/` 下的稳定路径，并 MUST 不替代 OpenSpec requirements、README quickstart、AGENTS 操作规则或 project surface inventory 的审计解释职责。

#### Scenario: 索引文件可定位
- **WHEN** AI agent 或维护者准备进行非平凡改动
- **THEN** 项目 MUST 提供 `docs/maintainer_context_index.yaml` 或等价稳定路径
- **AND** 该索引 MUST 声明自身是机器可读治理索引，而不是运行时配置、训练配置或 OpenSpec requirement 全文

#### Scenario: 索引不成为运行时入口
- **WHEN** 用户运行训练、评估、预处理、诊断或本地产物清理命令
- **THEN** runtime MUST 不要求读取维护上下文索引
- **AND** 缺少本地数据、checkpoint、cache 或 outputs MUST 不影响索引读取和验证

### Requirement: 索引覆盖 AI 任务路由
维护上下文索引 SHALL 记录常见改动类型到上下文读取顺序、主要修改区域和验证命令的映射。任务路由 MUST 至少覆盖模型/forward/registry、数据与 batch contract、配置和 virtual config、CLI/脚本入口、诊断/viewer、输出产物/cache、OpenSpec artifact 和文档生命周期改动。

#### Scenario: 模型改动可从索引定位上下文
- **WHEN** AI agent 需要新增或修改模型、forward 输出或 registry 暴露
- **THEN** 索引 MUST 指向相关 OpenSpec capability、`src/kd_sensing/models/`、registry/default component、shared batch/runtime 和 focused tests
- **AND** 索引 MUST 区分 config-only baseline、component baseline、whole-model exception 和 workflow/paper reproduction

#### Scenario: CLI 或配置改动可从索引定位治理表
- **WHEN** AI agent 需要新增 CLI、脚本、root config、experiment config 或 virtual config
- **THEN** 索引 MUST 指向 pyproject console scripts、`src/kd_sensing/cli/`、`scripts/` allowlist、配置 lifecycle 和对应验证命令
- **AND** 索引 MUST 提醒不得恢复 retired route、旧兼容 wrapper 或退役实体 YAML

### Requirement: 索引覆盖机器可读治理表
维护上下文索引 SHALL 保存可被测试消费的治理表。首批治理表 MUST 至少覆盖 Python 脚本入口事实、root fusion config allowlist、模型注册 allowlist、batch/runtime 分支 allowlist、热点 symbol/file budgets、快速健康检查命令和退役路线 token。固定 shell orchestration 不再作为 current allowlist 维护。

#### Scenario: 测试读取入口 allowlist
- **WHEN** 架构边界测试检查 `scripts/`、`tools/analysis/` 或 package CLI 入口
- **THEN** 测试 MUST 能从维护上下文索引读取允许入口及其 lifecycle
- **AND** 新增入口缺少索引登记时测试 MUST 失败或给出明确修复信息

#### Scenario: 测试读取模型和热点治理表
- **WHEN** 架构边界测试检查新增整模型注册、batch/runtime 分支或热点预算
- **THEN** 测试 MUST 能从维护上下文索引读取对应 allowlist 或 budget
- **AND** 缺少 whole-model exception、budget 或明确登记时测试 MUST 失败

### Requirement: 索引 schema 可轻量验证
项目 SHALL 提供维护上下文索引的轻量 schema 或等价验证逻辑。验证 MUST 检查必填 section、已登记路径存在性、lifecycle 值合法性、列表项唯一性和关键命令使用 `kd_mm_beam` 环境。

#### Scenario: 索引缺少必填 section
- **WHEN** `docs/maintainer_context_index.yaml` 缺少任务路由、治理表、健康检查命令或退役路线 section
- **THEN** 架构边界或文档健康检查 MUST 失败
- **AND** 失败信息 MUST 指向缺失 section 和预期字段

#### Scenario: 索引 lifecycle 值非法
- **WHEN** 索引中的 entrypoint、capability 或文档 lifecycle 使用未知值
- **THEN** 验证 MUST 失败
- **AND** 失败信息 MUST 列出允许值或指向对应 OpenSpec lifecycle 分类

### Requirement: 索引与权威来源对齐
维护上下文索引 SHALL 与 AGENTS、AI 维护导航、project surface inventory、OpenSpec specs、pyproject 和源码文件存在性保持一致。索引 MAY 摘要这些来源的路径和分类，但 MUST 不覆盖 OpenSpec requirement 或 README quickstart 的当前推荐入口判断。

#### Scenario: 索引引用不存在文件
- **WHEN** 索引登记的源码、脚本、配置、文档或 OpenSpec spec 路径不存在
- **THEN** 架构边界或文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求删除误登记项、恢复文件或更新索引路径

#### Scenario: OpenSpec capability 未登记 lifecycle
- **WHEN** `openspec/specs/<capability>/spec.md` 存在但索引或 inventory 未提供 lifecycle 分类
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求补充 `current`、`supporting` 或 `retired-tombstone` 分类

### Requirement: 索引变更无运行副作用
维护上下文索引能力 SHALL 只影响文档、OpenSpec artifact、测试治理数据和静态健康检查。实现该能力 MUST 不改变训练、评估、预处理、模型 forward、数据 split、配置解析、checkpoint schema、输出目录或本地产物清理语义。

#### Scenario: 实现索引不改变 runtime
- **WHEN** 本 change 完成
- **THEN** 项目 MUST 不新增长期训练/评估/预处理 CLI
- **AND** 项目 MUST 不修改默认 dataset 读取、模型构建、metric 计算、checkpoint 写出或 runtime output 分区

#### Scenario: 索引验证不读取本地产物
- **WHEN** 开发者运行索引相关健康检查
- **THEN** 检查 MUST 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event
- **AND** 检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact、pyproject 和测试文件

### Requirement: Entrypoint owner metadata
维护上下文索引 SHALL 为长期保留 entrypoint 记录 owner metadata。每个 package CLI、research diagnostic、dataset preparation、config generator 和 local/manual helper entry MUST 记录 owner module 或 owner script、responsibility、output boundary 和 lifecycle；Python thin alias 与固定 GPU shell 不再属于当前 entrypoint lifecycle。

#### Scenario: entrypoint metadata 完整
- **WHEN** entrypoint 出现在维护上下文索引
- **THEN** entry MUST 包含 lifecycle、owner module 或 owner script、responsibility 和 output boundary
- **AND** output boundary MUST 表明 read-only、ignored outputs/logs/cache、dataset preparation target 或显式用户路径

#### Scenario: retired route guard 可审计
- **WHEN** entrypoint 名称、owner module 或参数容易与退役路线混淆
- **THEN** 索引 MUST 记录 retired route guard 或 caveat
- **AND** inventory MUST 保留人类可读解释

### Requirement: package CLI 索引双向同步
维护上下文索引 SHALL 将 package CLI 视为 pyproject console scripts 的机器可读分类，而不是单向备注。索引中的 package CLI 集合 MUST 与 `pyproject.toml` 的 `[project.scripts]` 保持双向一致。

#### Scenario: package CLI 完整登记
- **WHEN** 项目声明 package console script
- **THEN** 维护上下文索引 MUST 登记该 script 的 name、target 和 lifecycle
- **AND** lifecycle MUST 属于索引允许的 entrypoint lifecycle values

#### Scenario: 删除 CLI 同步索引
- **WHEN** 某 package console script 从 pyproject 删除
- **THEN** 维护上下文索引 MUST 同步删除或重新分类该入口
- **AND** 架构边界测试 MUST 不允许 stale package CLI 登记长期存在

### Requirement: Hotspot budget 行动元数据
维护上下文索引 SHALL 为 hotspot file budgets 和 symbol budgets 提供机器可读行动元数据。每个 budget entry MUST 记录 priority、status、rationale、validation commands 和至少一种后续动作线索；后续动作线索 MAY 是 split targets、consolidation targets、accepted-size rationale 或推荐 next change。索引 MUST 能表达硬预算、软预算、理由化例外、合并候选和右尺寸化接受状态。

#### Scenario: hotspot entry 包含行动字段
- **WHEN** `docs/maintainer_context_index.yaml` 登记 file 或 symbol budget
- **THEN** 每个 entry MUST 包含 `priority`、`status`、`rationale` 和 `validation_commands`
- **AND** entry MUST 包含 `split_targets`、`consolidation_targets`、`accepted_size_rationale` 或 `next_change` 中至少一种后续动作线索
- **AND** `priority`、`status` 和 enforcement 相关字段 MUST 使用索引声明的允许值

#### Scenario: Codex 可从索引定位下一步
- **WHEN** AI agent 读取 hotspot budget
- **THEN** 索引 MUST 提供足以判断下一步应拆分、合并、监控、接受当前尺寸或调整预算的机器可读字段
- **AND** 详细 caveat 可继续由 `docs/project_surface_inventory.md` 提供

#### Scenario: facade 与业务热点 enforcement 可区分
- **WHEN** 架构边界测试读取热点预算
- **THEN** 索引 MUST 能区分 `facade-budget` 或等价硬预算状态与 `monitor`、`defer-with-rationale`、`right-size-accepted`、`merge-candidate` 等非硬预算状态
- **AND** facade 硬预算超限 MUST 被判为失败，业务热点 headroom 内的理由化超限 MAY 被接受

#### Scenario: 合并候选有明确 owner
- **WHEN** hotspot entry 被标记为 `merge-candidate` 或包含 `consolidation_targets`
- **THEN** 索引 MUST 记录合并目标、owner module 或足够明确的 owner 说明
- **AND** validation commands MUST 覆盖合并后可能受影响的公开契约

#### Scenario: right-size accepted 不替代验证
- **WHEN** hotspot entry 被标记为 `right-size-accepted` 或登记 accepted-size rationale
- **THEN** 索引 MUST 记录为什么保持当前尺寸比继续拆分更可维护
- **AND** entry MUST 保留 focused validation commands，防止该状态被解释为永不重构的永久豁免

### Requirement: Hotspot metadata 不替代 inventory 解释
维护上下文索引中的 hotspot metadata SHALL 作为机器可读行动摘要。inventory MUST 继续保留热点原因、暂缓解释和审计上下文；二者看似冲突时 MUST 被视为治理漂移。

#### Scenario: inventory 提供长解释
- **WHEN** hotspot budget 在索引中登记
- **THEN** `docs/project_surface_inventory.md` MUST 继续包含该路径或 symbol 的解释性条目
- **AND** 索引 `rationale` MUST 是短摘要，不得替代 inventory 的审计说明

### Requirement: Hotspot remediation wave metadata
维护上下文索引 SHALL 能为高风险热点修复记录 remediation wave metadata。每个 wave entry MUST 指明 wave id、目标路径、owner module、planned action、public surface policy、validation commands 和 rollback note。planned action MUST 能表达 split、consolidate、keep-and-test、owner-facade、hard-budget 或 accepted-size。

#### Scenario: wave metadata 可定位实施范围
- **WHEN** AI agent 或架构测试读取维护上下文索引
- **THEN** 索引 MUST 能列出当前 change 涉及的热点 wave、目标源码路径和 owner module
- **AND** 每个 wave MUST 有 planned action 和 focused validation commands

#### Scenario: public surface policy 防止误删入口
- **WHEN** wave 触碰已登记 CLI、public import owner 或 baseline reproduction module
- **THEN** 索引 MUST 记录 public surface policy，例如 keep-public-import、thin-owner、no-public-surface 或 remove-internal-only
- **AND** 架构测试或 focused tests MUST 能验证公开入口没有被意外删除

#### Scenario: keep-and-test 是合法动作
- **WHEN** hotspot 审核发现某个模块规模较小、职责内聚且继续拆分会增加跳转成本
- **THEN** 索引 MAY 将 planned action 记录为 keep-and-test
- **AND** entry MUST 说明保留理由和对应 focused tests

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

### Requirement: 维护索引记录本次支持面收敛结果
维护上下文索引 SHALL 记录本次删减后的 entrypoint、hotspot、merge-candidate、dependency 和 remediation wave 状态。索引 MUST 将 package console scripts 作为当前入口事实，MUST 不继续登记已删除的 Python thin alias 为 current entrypoint。

#### Scenario: entrypoint 索引不保留 thin alias
- **WHEN** `scripts/train.py`、`scripts/evaluate.py`、`scripts/preprocess.py` 或 BeamBench thin alias 从源码删除
- **THEN** 维护索引 MUST 删除或重新分类对应 script entry
- **AND** package CLI 索引 MUST 与 `pyproject.toml` 的 `[project.scripts]` 保持双向一致

#### Scenario: 删除和合并候选有行动元数据
- **WHEN** 索引记录 `communication_state_features`、LiDAR pillar 原型、dataset runtime adapter 框架或重复 `OutputRegistry` 的收敛状态
- **THEN** entry MUST 标明 planned action、public surface policy、validation commands 和 rollback note
- **AND** public surface policy MUST 能区分 `remove-internal-only`、`merge-into-owner` 和 `keep-public-import`

#### Scenario: CSI hardening matrix 分类更新
- **WHEN** CSI hardening 配置矩阵从重复实体 YAML 收敛为 base+overlay 或 recipe
- **THEN** 索引或 inventory MUST 记录 base config、overlay/recipe 位置、当前配置 ID 范围和验证命令
- **AND** 架构边界测试 MUST 不再要求每个矩阵 ID 都对应一份完整实体 YAML

#### Scenario: dev dependency audit 可追踪
- **WHEN** dev extra 删除未使用依赖
- **THEN** 维护索引或 inventory MUST 记录该删除不影响 runtime dependencies
- **AND** 若后续重新引入同类依赖，必须在对应 change 中说明当前使用点和验证命令
