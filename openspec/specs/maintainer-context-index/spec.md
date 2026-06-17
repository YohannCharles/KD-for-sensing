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
维护上下文索引 SHALL 保存可被测试消费的治理表。首批治理表 MUST 至少覆盖 Python 脚本入口 allowlist、shell orchestration allowlist、root fusion config allowlist、模型注册 allowlist、batch/runtime 分支 allowlist、热点 symbol/file budgets、快速健康检查命令和退役路线 token。

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
维护上下文索引 SHALL 为长期保留 entrypoint 记录 owner metadata。每个 package CLI、thin alias、research diagnostic、dataset preparation 和 shell orchestration entry MUST 记录 owner module 或 owner script、responsibility、output boundary 和 lifecycle。

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
维护上下文索引 SHALL 为 hotspot file budgets 和 symbol budgets 提供机器可读行动元数据。每个 budget entry MUST 记录 priority、status、split targets、rationale 和 validation commands，且 MAY 记录推荐 next change。

#### Scenario: hotspot entry 包含行动字段
- **WHEN** `docs/maintainer_context_index.yaml` 登记 file 或 symbol budget
- **THEN** 每个 entry MUST 包含 `priority`、`status`、`split_targets`、`rationale` 和 `validation_commands`
- **AND** `priority` 和 `status` MUST 使用索引声明的允许值

#### Scenario: Codex 可从索引定位下一步
- **WHEN** AI agent 读取 hotspot budget
- **THEN** 索引 MUST 提供足以定位下一步拆分方向的 `split_targets` 或 `next_change`
- **AND** 详细 caveat 可继续由 `docs/project_surface_inventory.md` 提供

### Requirement: Hotspot metadata 不替代 inventory 解释
维护上下文索引中的 hotspot metadata SHALL 作为机器可读行动摘要。inventory MUST 继续保留热点原因、暂缓解释和审计上下文；二者看似冲突时 MUST 被视为治理漂移。

#### Scenario: inventory 提供长解释
- **WHEN** hotspot budget 在索引中登记
- **THEN** `docs/project_surface_inventory.md` MUST 继续包含该路径或 symbol 的解释性条目
- **AND** 索引 `rationale` MUST 是短摘要，不得替代 inventory 的审计说明

