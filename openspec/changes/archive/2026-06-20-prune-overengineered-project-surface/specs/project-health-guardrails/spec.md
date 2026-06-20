## ADDED Requirements

### Requirement: 健康护栏使用最小结构化来源
项目健康护栏 MUST 优先验证权威来源本身，包括 `pyproject.toml`、真实源码路径、OpenSpec requirements、当前 README/docs 中的路径或命令、轻量 import probes 和小型 lifecycle inventory。健康护栏 MUST NOT 要求维护一个完整镜像源码结构、入口 allowlist、热点预算和文档路由的长期 YAML，除非该 YAML 被明确保留为最小 inventory。

#### Scenario: pyproject 脚本直接验证
- **WHEN** 架构边界测试检查 console scripts
- **THEN** 测试 MUST 直接读取 `pyproject.toml` 的 `[project.scripts]`
- **AND** 测试 MUST 不要求同一脚本清单在维护上下文索引中重复登记

#### Scenario: 热点事实从小型来源验证
- **WHEN** 架构边界测试检查热点、facade 或 current entrypoint 回流
- **THEN** 测试 MUST 使用 OpenSpec、项目表面积 inventory、真实文件路径或少量测试常量中的稳定事实
- **AND** 测试 MUST 不要求维护完整源码目录清单或大段 YAML schema projection logic

### Requirement: 退役路线护栏不依赖单一索引
项目健康护栏 MUST 继续拒绝 retired route 以 CLI、配置、registry 名称、facade、script 或 quickstart wording 回流，但该护栏 MUST 不依赖 `docs/maintainer_context_index.yaml` 的存在。退役 token 和禁止入口 MAY 存在于 OpenSpec requirements、project surface inventory 或 focused tests 中。

#### Scenario: retired route 被写成当前入口
- **WHEN** README、current docs、OpenSpec current specs、pyproject、configs 或 registry 把退役路线登记为 current quickstart、root config、console script 或长期 workflow
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改为 retired/supporting/migration guard 语义或删除该入口

#### Scenario: migration guard 合法引用被允许
- **WHEN** docs 或 specs 只在 migration guard、历史说明、拒绝边界或 retired tombstone 中提到退役路线
- **THEN** 健康检查 MUST 允许该引用
- **AND** 检查 MUST 不把合法拒绝说明误判为入口回流

## MODIFIED Requirements

### Requirement: 架构边界测试验证结构化事实而非 prose mirror
项目健康护栏 MUST 验证长期稳定事实，例如入口路径、console script、lifecycle、配置引用、轻量导入边界、退役 token 和本地产物边界。事实来源 MAY 是 OpenSpec requirements、project surface inventory、pyproject、AST/path/import 扫描或小型测试常量；架构边界测试 MUST 不逐字镜像 README、docs 或 OpenSpec 的自然语言段落，也不 MUST 通过大型维护上下文索引间接验证可直接读取的事实。

#### Scenario: 文档自然语言改写不触发结构测试失败
- **WHEN** README 或 docs 在不改变入口、路径、lifecycle、命令、配置引用或退役语义的情况下改写说明文字
- **THEN** 架构边界测试 MUST 不因固定短语缺失而失败
- **AND** 测试 MUST 继续验证路径、命令、OpenSpec lifecycle 或退役语义是否一致

#### Scenario: 当前入口事实仍被验证
- **WHEN** README、docs、OpenSpec 或 current inventory 声明当前 CLI、配置路径、dataset type、模型注册名或诊断入口
- **THEN** 架构边界测试 MUST 验证对应路径、pyproject entry point 或源码 owner 存在
- **AND** stale 当前入口引用 MUST 失败

#### Scenario: 退役 wording guard 保留
- **WHEN** current docs 或 current specs 将已退役路线写成 quickstart、active mainline、默认 workflow 或长期入口
- **THEN** 健康护栏 MUST 继续失败
- **AND** 失败信息 MUST 指向加入退役限定、更新 lifecycle 或删除推荐入口

#### Scenario: 护栏检查无运行副作用
- **WHEN** 开发者运行架构边界测试或文档健康检查
- **THEN** 检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact、pyproject 和测试文件
- **AND** 检查 MUST 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event

### Requirement: Dataset contract helper 热点治理
项目健康护栏 MUST 鼓励 DeepSense6G dataset contract helper 拆分，并防止新的契约规则继续堆入 `DeepSense6GDataset` 超长类。热点治理 MAY 记录在 project surface inventory、OpenSpec tasks 或 focused tests 中；项目不再 MUST 通过维护上下文索引记录 helper 拆分方向和预算。

#### Scenario: DeepSense6GDataset 预算下降或保持有理由
- **WHEN** helper 拆分完成
- **THEN** `docs/project_surface_inventory.md`、OpenSpec tasks 或 focused tests MUST 记录保留职责、拆分方向或暂缓原因
- **AND** 不得为了记录预算而强制恢复 `docs/maintainer_context_index.yaml`

#### Scenario: 新契约规则进入 helper
- **WHEN** 后续新增 GPS feature mode、beam target source、column guard 或 cache path rule
- **THEN** 主要实现 MUST 位于 DeepSense6G contract helper 模块
- **AND** 架构或 focused tests MUST 防止这些规则继续扩大 dataset class 主体

### Requirement: JEPA benchmark facade 和窄模块预算
项目健康护栏 MUST 防止 JEPA benchmark facade 重新变厚。若 facade 被保留，它 MUST 只委托窄 owner 模块；若本 change 删除 facade，当前 CLI、docs 和 tests MUST 直接指向保留的 owner 模块或正式入口。窄模块职责和预算 MAY 记录在 project surface inventory、OpenSpec tasks 或 focused tests 中，不再 MUST 登记到维护上下文索引。

#### Scenario: facade 超预算失败
- **WHEN** 架构边界测试扫描保留的 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`
- **THEN** 文件行数或导入职责 MUST 保持薄 facade 范围
- **AND** 超预算时测试 MUST 要求继续拆分到窄模块或删除 facade，而不是扩大 facade

#### Scenario: 新窄模块登记职责
- **WHEN** 拆分新增 JEPA benchmark 内部模块
- **THEN** project surface inventory、OpenSpec tasks 或 focused tests MUST 说明模块职责和防回流边界
- **AND** 不得为了登记该模块而强制维护完整上下文索引

### Requirement: CLI 和脚本入口健康检查
项目健康护栏 MUST 检查 CLI 和 scripts 入口不变厚。检查 MUST 基于 `pyproject.toml`、真实脚本路径、current docs、project surface inventory 或 focused tests 中的最小入口事实，拒绝未登记 current 入口、恢复 Python thin alias 和明显复制 workflow 逻辑的脚本。

#### Scenario: 新脚本缺少 owner module
- **WHEN** `scripts/`、`tools/analysis/` 或 package CLI 新增 current 入口
- **THEN** 变更 MUST 在 project surface inventory、README/docs 或 OpenSpec tasks 中登记 owner module、responsibility 和 output boundary
- **AND** 缺少登记时架构边界测试 MUST 失败

#### Scenario: 脚本入口包含训练循环 marker
- **WHEN** 保留的 `scripts/` research diagnostic、dataset preparation 或 shell orchestration 包含大段训练循环、模型 forward、optimizer step 或重复 package CLI 主逻辑
- **THEN** 健康检查 MUST 失败或要求重新分类为 owner module
- **AND** 修复路径 MUST 是委托包内实现或创建正式 package module

## REMOVED Requirements

### Requirement: 健康护栏验证维护上下文索引
**Reason**: 维护上下文索引已经变成独立维护对象，和 pyproject、OpenSpec、inventory、真实文件路径重复，增加同步成本。
**Migration**: 使用 `project-health-guardrails` 中的“健康护栏使用最小结构化来源”和“退役路线护栏不依赖单一索引”要求。

#### Scenario: 索引缺失不再单独失败
- **WHEN** `docs/maintainer_context_index.yaml` 被删除或收缩为非必需文件
- **THEN** 架构边界测试 MUST 不仅因该文件缺失而失败
- **AND** 关键入口、退役路线和轻量导入事实 MUST 由其它权威来源验证

### Requirement: 健康护栏从索引读取治理 allowlist
**Reason**: 长期 allowlist 放在维护索引中会复制真实文件树和 pyproject，清理时需要同步多份事实。
**Migration**: 测试直接读取 pyproject、真实路径、OpenSpec、project surface inventory 或小型测试常量。

#### Scenario: allowlist 来源迁移
- **WHEN** 架构边界测试检查 scripts、tools、配置、模型或 hotspot
- **THEN** 测试 MUST 不要求维护上下文索引作为唯一 allowlist 来源
- **AND** 未登记 current 入口仍 MUST 被 focused checks 拒绝

### Requirement: 索引一致性检查不放宽退役路线护栏
**Reason**: 退役路线护栏需要保留，但不应依赖维护索引存在。
**Migration**: 使用“退役路线护栏不依赖单一索引”要求，通过 OpenSpec、inventory 和 focused tests 拒绝回流。

#### Scenario: 无索引时仍拒绝回流
- **WHEN** 维护上下文索引不存在
- **THEN** retired route 作为 current CLI、配置、registry 或 quickstart 回流仍 MUST 失败
- **AND** 合法历史说明仍 MUST 被允许

### Requirement: 维护上下文索引测试 helper 私有化
**Reason**: 删除或收缩维护索引后，专门的测试 helper 不再有长期价值。
**Migration**: 将仍需要的解析逻辑内联到 focused tests 或迁移到更小的测试私有 helper，且不得成为 runtime API。

#### Scenario: helper 删除
- **WHEN** 测试不再读取维护上下文索引
- **THEN** `tests/helpers/maintainer_context.py` MAY 删除
- **AND** runtime MUST 继续不导入测试 helper

### Requirement: pyproject 和 maintainer index 双向一致
**Reason**: console script 的权威来源是 `pyproject.toml`，要求第二份索引双向一致属于重复治理。
**Migration**: 架构边界测试直接读取 `pyproject.toml`，并与 current docs 或 CLI smoke 需要的入口核对。

#### Scenario: pyproject 直接检查
- **WHEN** `[project.scripts]` 新增、删除或重命名 `kd-sensing-*` console script
- **THEN** 测试 MUST 基于 pyproject 和 current docs 判断是否有效
- **AND** 不得要求同步维护上下文索引条目

### Requirement: 验证 hotspot 行动元数据
**Reason**: 维护上下文索引中的 hotspot action metadata 太细，容易变成与 tasks、inventory 和测试重复的治理表。
**Migration**: 高风险 wave 的目标、验证命令和回滚条件记录在 active change tasks、design 或 project surface inventory 中。

#### Scenario: hotspot metadata 迁出索引
- **WHEN** 一个高风险源码表面修复 wave 开始
- **THEN** tasks 或实现说明 MUST 记录目标文件、owner 边界和 focused validation commands
- **AND** 测试 MUST 不要求这些字段存在于维护上下文索引
