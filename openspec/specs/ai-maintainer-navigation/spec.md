# ai-maintainer-navigation Specification

## Purpose
定义面向 AI agent 和维护者的轻量导航层，用于在非平凡改动前快速判断项目权威来源、任务路由、常见误读边界和验证命令，同时避免替代 README、AGENTS、OpenSpec specs 或项目表面积 inventory。
## Requirements
### Requirement: AI 维护导航文档存在且职责清晰
项目 SHALL 提供一份面向 AI agent 和维护者的导航文档，用于在非平凡代码或文档改动前快速判断权威来源、当前状态、任务路由、误读边界和验证命令。该文档 MUST 保持为薄导航层，不得替代 README 的 quickstart、OpenSpec specs 的需求契约或项目表面积 inventory 的完整审计职责。

#### Scenario: 导航文档可定位
- **WHEN** 开发者或 AI agent 阅读项目操作规则
- **THEN** `AGENTS.md` MUST 指向 AI 维护导航文档
- **AND** 导航文档 MUST 位于 `docs/` 下的稳定 Markdown 路径

#### Scenario: 导航不重复完整目录清单
- **WHEN** 开发者阅读 AI 维护导航文档
- **THEN** 文档 MUST 描述阅读顺序、任务路由和边界判断
- **AND** 文档 MUST NOT 维护完整源码目录清单或替代 OpenSpec requirement 内容

### Requirement: 权威来源优先级明确
AI 维护导航文档 SHALL 明确修改前的权威来源优先级。优先级 MUST 至少覆盖用户当前请求、AGENTS 操作规则、active OpenSpec change、当前 `openspec/specs/`、README/docs workflow、源码与测试、OpenSpec archive、历史报告和本地产物。

#### Scenario: 多来源冲突时有优先级
- **WHEN** README、OpenSpec archive、当前 specs、active change 或本地产物给出看似冲突的信息
- **THEN** AI 维护导航文档 MUST 说明按当前请求、AGENTS、active change、当前 specs、README/docs、源码测试、历史/archive/本地产物的顺序判断
- **AND** 文档 MUST 明确 archive 和历史报告不能作为当前支持契约覆盖当前 specs

#### Scenario: active change 状态需要显式检查
- **WHEN** 仓库存在 active OpenSpec change
- **THEN** 导航文档 MUST 要求通过 `openspec list --json` 和 `openspec status --change <change>` 或等价命令判断状态
- **AND** 文档 MUST 提醒已完成但未归档的 change 仍可能影响当前工作树解释

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

### Requirement: 任务路由表覆盖常见改动类型
AI 维护导航文档 SHALL 提供任务路由表，帮助维护者从变更类型映射到先读文档、主要修改区域和验证命令。路由表 MUST 覆盖模型/forward、数据与 batch contract、配置和 virtual config、CLI/脚本入口、输出产物/cache、诊断/viewer、OpenSpec artifact 和文档生命周期改动。

#### Scenario: 修改模型时有路由
- **WHEN** 开发者计划新增或修改模型、forward 输出或 registry 暴露
- **THEN** 导航文档 MUST 指向模型相关 OpenSpec、`src/kd_sensing/models/`、registry/default component 边界和 forward/config focused tests

#### Scenario: 修改数据契约时有路由
- **WHEN** 开发者计划新增或修改 dataset 字段、batch key、模态输入或 target 语义
- **THEN** 导航文档 MUST 指向 dataset/modality contract specs、`src/kd_sensing/data/`、`src/kd_sensing/engine/batch.py`、shared runtime 和相关 focused tests

#### Scenario: 修改配置或入口时有路由
- **WHEN** 开发者计划新增配置、virtual config、CLI、脚本或 workflow 入口
- **THEN** 导航文档 MUST 指向配置生命周期、`pyproject.toml`、`src/kd_sensing/cli/`、`scripts/` allowlist、inventory 和 CLI/config/architecture boundary checks

### Requirement: 常见误读边界被显式列出
AI 维护导航文档 SHALL 列出项目中容易误读的路径和状态。误读清单 MUST 至少覆盖 generated metadata、ignored runtime artifacts、本地数据、OpenSpec archive、retired research lines、virtual configs、active change 状态和当前打开文件不等于项目权威入口。

#### Scenario: generated metadata 不作为源码权威
- **WHEN** AI agent 当前打开 `src/kd_sensing.egg-info/SOURCES.txt`、`entry_points.txt` 或其它 packaging metadata
- **THEN** 导航文档 MUST 明确这些文件是 generated metadata
- **AND** 文档 MUST 指向 `pyproject.toml`、`src/kd_sensing/`、README 和 OpenSpec 作为结构与入口判断来源

#### Scenario: ignored runtime artifacts 不作为支持面
- **WHEN** 工作树包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.pytest_cache` 或 Python bytecode
- **THEN** 导航文档 MUST 明确这些路径默认属于本地输入或运行产物
- **AND** 文档 MUST 说明它们不得自动纳入源码变更或作为当前支持入口证据

#### Scenario: retired 与 virtual config 边界清晰
- **WHEN** 开发者遇到旧 KD、HiST、Top8、residual、camera residual、Raymobtime s008 或不存在实体 YAML 的 virtual config 路径
- **THEN** 导航文档 MUST 提醒先查 README、inventory 和 config specs
- **AND** 文档 MUST 明确不得用兼容 wrapper 或实体 YAML 恢复已退役路线

### Requirement: 导航文档纳入生命周期与健康检查
项目 SHALL 将 AI 维护导航文档纳入文档生命周期分类和架构边界检查。测试 MUST 能在不读取真实数据、不加载 checkpoint、不启动训练的情况下验证导航文档存在、关键标记齐全，并与 AGENTS 和 inventory 的引用保持一致。

#### Scenario: inventory 分类导航文档
- **WHEN** 开发者阅读 `docs/project_surface_inventory.md`
- **THEN** inventory MUST 将 AI 维护导航文档分类为当前 agent/maintainer navigation 或等价生命周期
- **AND** inventory MUST 说明它不替代 README、AGENTS 或 OpenSpec specs

#### Scenario: 架构边界检查导航文档
- **WHEN** 开发者运行架构边界测试
- **THEN** 测试 MUST 验证 AI 维护导航文档存在并包含权威来源、任务路由、generated metadata、ignored runtime artifacts、virtual config、retired research line 和 `kd_mm_beam` 等关键标记
- **AND** 测试 MUST 验证 `AGENTS.md` 和 inventory 对导航文档的引用或分类存在

### Requirement: 导航变更无运行时副作用
AI 维护导航能力 SHALL 只改变文档和健康检查，不得改变训练、评估、预处理、模型 forward、数据 split、配置解析、运行输出或本地产物清理语义。

#### Scenario: 实现导航文档不改变 runtime
- **WHEN** 本 change 实现完成
- **THEN** 项目 MUST 不新增长期训练/评估/预处理 CLI
- **AND** 项目 MUST 不修改默认训练输出目录、checkpoint schema、dataset 读取语义或模型构建语义

#### Scenario: 不自动清理本地产物
- **WHEN** 本 change 实现完成
- **THEN** 实现 MUST NOT 删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.egg-info` 或其它 ignored 本地产物
- **AND** 如需清理本地产物，仍 MUST 使用现有 manifest 或显式确认流程

### Requirement: 模型架构扩展导航
AI 维护导航文档 MUST 在模型、forward、registry 或 baseline 改动路由中指向 `model-architecture-extension-contract`、`modular-sequence-model`、`component-registry`、共享 batch/runtime 和相关 focused tests。AI agent MUST 在非平凡模型改动前判断改动属于 config-only、component baseline、whole-model exception 还是 workflow/paper reproduction。

#### Scenario: AI 新增普通 baseline 前检查契约
- **WHEN** AI agent 准备新增或修改普通 supervised/adaptation baseline
- **THEN** 导航文档 MUST 要求优先选择 `modular_sequence` 配置或子组件注册路径
- **AND** agent MUST 不直接新增整模型注册名，除非 change artifact 明确 whole-model exception

#### Scenario: AI 新增论文复现 workflow 前检查边界
- **WHEN** AI agent 准备新增 paper reproduction 或多阶段 baseline workflow
- **THEN** 导航文档 MUST 指向 `src/kd_sensing/baselines/<family>/`、包内 CLI、脚本 allowlist 和本地产物边界
- **AND** agent MUST 不复制通用训练循环或新增 root-level 旧式入口

### Requirement: 模型改动验证路由更具体
模型相关任务路由 MUST 区分模块化组件、整模型例外、batch metadata 和 workflow baseline 的验证命令。至少 MUST 提到架构边界测试、对应模型 focused forward tests、配置加载 characterization，以及触碰 reliability metadata 时的 difficulty/batch tests。

#### Scenario: reliability-aware 模型改动验证
- **WHEN** 模型改动声明消费 observability/reliability metadata
- **THEN** 导航文档 MUST 建议运行相关 batch/difficulty focused tests
- **AND** 验证说明 MUST 覆盖普通 baseline 忽略 metadata 与 opt-in 模型接收 metadata 两种路径

### Requirement: 导航优先读取维护上下文索引
AI 维护导航文档 SHALL 将中心化维护上下文索引纳入非平凡改动前的当前状态检查顺序。导航文档 MUST 要求 agent 先通过索引定位任务路由、治理表和验证命令，再按需读取 README、project surface inventory、OpenSpec specs 和源码。

#### Scenario: 非平凡改动前读取索引
- **WHEN** AI agent 计划修改模型、数据契约、配置、CLI、诊断 workflow、OpenSpec artifact 或文档生命周期
- **THEN** `docs/agent_navigation.md` MUST 指向维护上下文索引
- **AND** 导航文档 MUST 说明索引用于快速定位上下文，不替代 OpenSpec requirements 或 README quickstart

#### Scenario: 当前打开文件不覆盖索引路由
- **WHEN** IDE 当前打开文件是薄 CLI alias、generated metadata、测试 allowlist 或本地输出摘要
- **THEN** 导航文档 MUST 要求 agent 使用维护上下文索引确认该文件所属 lifecycle 和任务路由
- **AND** agent MUST 不把当前打开文件单独视为项目权威入口

### Requirement: 导航说明索引与 inventory 的职责边界
AI 维护导航文档 SHALL 说明维护上下文索引与 `docs/project_surface_inventory.md` 的职责边界。索引 MUST 被描述为机器可读的治理事实入口，inventory MUST 被描述为解释性审计和历史上下文来源。

#### Scenario: 读取 lifecycle 时知道事实来源
- **WHEN** AI agent 需要判断某 capability、entrypoint、config 或热点是否属于当前支持面
- **THEN** 导航文档 MUST 指向维护上下文索引中的结构化分类
- **AND** 导航文档 MUST 指向 inventory 或对应 OpenSpec spec 以理解分类原因和 caveat

#### Scenario: 文档表格与索引冲突时有处理方式
- **WHEN** inventory、README、导航文档和维护上下文索引之间出现看似冲突的分类或入口说明
- **THEN** 导航文档 MUST 要求把它视为治理漂移
- **AND** 后续变更 MUST 通过 OpenSpec change 同步索引、inventory 和对应 specs，而不是任选一处作为事实

### Requirement: 导航按右尺寸化决策处理热点
AI 维护导航文档 SHALL 指导 agent 和维护者在处理热点模块、长函数、长类、facade 或 helper 边界时使用右尺寸化决策矩阵。导航 MUST 明确拆分、合并、监控、接受当前尺寸和预算调整都是可能的有效动作，且 MUST 禁止把热点预算机械解释为“所有大文件都要拆”。

#### Scenario: agent 修改热点前先判断动作类型
- **WHEN** AI agent 准备修改已登记热点、接近预算的 workflow、dataset、diagnostic module 或 facade
- **THEN** 导航文档 MUST 要求先读取 `docs/maintainer_context_index.yaml` 中的 hotspot metadata
- **AND** agent MUST 判断本次变更属于拆分、合并/收敛、监控、接受当前尺寸、预算调整或源码窄修复中的哪一种

#### Scenario: facade 与业务模块使用不同判断
- **WHEN** agent 遇到公开 CLI/import facade 和真实业务 workflow 同时接近预算
- **THEN** 导航文档 MUST 要求 facade 继续按硬预算和防回流规则处理
- **AND** 业务 workflow MUST 按 rationale、headroom、validation commands 和调用边界判断是否拆分或保持线性流程

#### Scenario: 低价值抽象优先合并
- **WHEN** agent 发现单调用点包装类、只为减少行数的小 helper、重复 utils 聚合或无公开兼容价值的 facade
- **THEN** 导航文档 MUST 要求优先考虑合并回清晰 owner 或改为私有局部 helper
- **AND** agent MUST 不通过新增兼容包装层、旧入口或绕过 `src/kd_sensing` 包结构来完成合并

#### Scenario: 输出方案包含验证和风险
- **WHEN** agent 提出或执行热点右尺寸化变更
- **THEN** 方案或最终说明 MUST 写明选择拆分、合并、监控或接受当前尺寸的原因
- **AND** 方案 MUST 列出对应 focused validation commands，并说明不会读取真实 `dataset/` 或写入 ignored runtime artifacts

### Requirement: 导航支持高风险修复 campaign
AI 维护导航文档 SHALL 指导 agent 在用户明确接受高风险时使用 remediation wave 计划，而不是把多个热点重构混成单次不可定位的大改。导航 MUST 要求 agent 先确认 active OpenSpec change、读取维护上下文索引、列出 wave 顺序和每个 wave 的验证命令，再开始源码实施。

#### Scenario: 高风险请求转为 wave 计划
- **WHEN** 用户要求完整修复热点架构且明确可以接受高风险
- **THEN** agent MUST 将方案拆成多个 remediation waves
- **AND** 每个 wave MUST 标明目标文件、计划动作、保留或改变的 public surface、focused tests 和回滚/停止条件

#### Scenario: 当前打开文件不被孤立处理
- **WHEN** IDE 当前打开 `data_factory.py`、`sequences.py`、loss 或 model 文件
- **THEN** agent MUST 把这些文件放回维护上下文索引和 wave 计划判断
- **AND** agent MUST 不因为文件当前打开就默认拆分，也不得因为文件较小就忽略测试和 owner 边界

#### Scenario: 小模块作为 keep-and-test 样板
- **WHEN** agent 审核到小而内聚的 loss、model 或 helper 模块
- **THEN** 导航 MUST 允许选择 keep-and-test
- **AND** agent MUST 说明为什么不拆，以及需要补充或保留哪些 focused tests

