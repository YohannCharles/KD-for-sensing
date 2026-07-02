# project-hotspot-governance Specification

## Purpose
定义维护性热点、右尺寸化预算、remediation wave、accepted owner 和回流防护的治理规则，避免只按行数或文件数机械拆分源码。

## Requirements

### Requirement: 维护性热点 inventory
项目 MUST 维护一份可审计的维护性热点 inventory，记录已知超长模块、超长函数、超长类、兼容 facade、推荐拆分方向、合并/收敛方向、暂缓原因和右尺寸化预算策略。新增热点或热点显著扩大时，项目 MUST 更新 inventory、拆分到稳定窄模块、合并低价值边界，或登记有验证命令支撑的理由化例外。

#### Scenario: 已知热点被命名
- **WHEN** 开发者审阅项目健康 inventory
- **THEN** inventory MUST 记录当前已知热点的文件路径、符号名、热点类型、当前规模指标、推荐拆分方向、合并/收敛方向或接受当前尺寸的理由
- **AND** inventory MUST 包含训练主循环、DeepSense6G/MMW dataset、BeamBench Image AE+GPS workflow、run index、evaluation pass、batch preparation 和 manifest builder 等当前高维护成本区域

#### Scenario: 静态检查发现未登记热点
- **WHEN** 架构边界或健康检查发现新增超长函数、超长类或 facade 回流
- **THEN** 检查 MUST 失败或输出明确失败信息
- **AND** 失败信息 MUST 指向更新 inventory、拆分到窄模块、合并低价值边界或增加有理由例外这几种修复路径之一

#### Scenario: facade 预算继续硬失败
- **WHEN** 已登记为 facade 或公开兼容入口的模块超过硬预算，或重新承载已迁出的 suite-specific helper 实现
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求将实现移回职责明确的窄模块或删除不再需要的兼容 facade

#### Scenario: 业务热点允许理由化 headroom
- **WHEN** 已登记业务 workflow、dataset 或 diagnostic analysis 模块略超预算但处于索引声明的 headroom 内
- **THEN** 健康检查 MAY 接受该状态
- **AND** 索引和 inventory MUST 提供 rationale、validation commands 和后续动作分类，例如 `monitor`、`split-next`、`right-size-accepted` 或 `merge-candidate`

#### Scenario: 低价值边界被合并或登记为合并候选
- **WHEN** 维护者发现单调用点包装类、只为减少行数的 helper 模块、重复 `utils` 聚合或无公开兼容价值的 facade
- **THEN** 实现 MUST 将其合并回清晰 owner、改为私有局部 helper，或在热点 inventory 中登记为 `merge-candidate`
- **AND** 合并 MUST 不把实现重新堆回公开 facade 或绕过当前包结构

#### Scenario: 右尺寸化检查不改变 runtime
- **WHEN** 开发者运行架构边界或健康检查
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

### Requirement: 高风险源码表面修复按 wave 管理
项目 MAY 对热点模块执行高风险结构重构，但该重构 MUST 按 remediation wave 管理。每个 wave MUST 记录目标文件、owner 边界、公开 import/CLI 保持策略、focused validation commands 和回滚条件。高风险 wave MUST 不把训练数学语义、数据 split 语义、beam label 口径、checkpoint schema 或默认输出目录作为隐式变更。

#### Scenario: wave 开始前捕获 baseline
- **WHEN** 维护者开始一个高风险热点修复 wave
- **THEN** tasks 或实现说明 MUST 记录该 wave 的目标文件、当前热点规模、公开入口和最小 focused tests
- **AND** 若已有测试红点，说明 MUST 区分既有红点和本 wave 引入的新红点

#### Scenario: wave 完成后独立验证
- **WHEN** 一个 wave 完成源码移动、拆分或合并
- **THEN** 维护者 MUST 运行该 wave 对应的 focused tests 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 未运行的验证 MUST 在最终说明中记录原因和剩余风险

#### Scenario: 小而内聚模块不被强制拆分
- **WHEN** 模块低于热点阈值、职责内聚且无重复抽象或公开边界问题
- **THEN** 健康护栏 MUST NOT 要求仅因相邻热点修复而拆分该模块
- **AND** 维护者 MAY 只补测试或登记为 monitor

### Requirement: 架构边界测试必须右尺寸化
架构边界测试 MUST 验证长期结构事实，而不是复制完整维护索引、文档短语、脚本 allowlist 或 OpenSpec prose。测试 SHOULD 保持可读、可维护，并优先从权威来源直接读取事实：`pyproject.toml`、真实路径、OpenSpec lifecycle inventory、tracked files、AST/import probes 和小型 retired token 常量。

#### Scenario: 删除大型治理镜像
- **WHEN** `tests/test_architecture_boundaries.py` 维护与 pyproject、inventory、README 或 OpenSpec 重复的长 allowlist
- **THEN** 本 change MUST 删除该镜像或改为从权威来源直接推导
- **AND** 测试 MUST 不要求维护完整源码目录清单、完整 package CLI 数据库或完整 hotspot budget 表

#### Scenario: 保留结构性失败
- **WHEN** 当前 docs 或 specs 引用不存在的 current config、console script、module path 或 capability lifecycle
- **THEN** 架构边界测试 MUST 继续失败
- **AND** 失败信息 MUST 指向修正文档、恢复文件或更新 lifecycle 分类，而不是放宽测试

### Requirement: 健康护栏变更必须有 focused 自检
重写健康护栏时 MUST 留下最小自检，证明它仍能拒绝三类关键回归：旧入口回流、tracked 本地产物进入源码、current path/config 引用失效。该自检 MUST 不读取真实数据、不启动训练、不写入运行产物。

#### Scenario: 架构边界 focused test
- **WHEN** 健康护栏重写完成
- **THEN** `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` MUST 通过
- **AND** 测试内容 MUST 覆盖 pyproject scripts、retired route guard、tracked artifact boundary 和 current config/path reference

#### Scenario: 本地 ignored cache 不影响测试
- **WHEN** 工作树存在 ignored `__pycache__`、`.pytest_cache`、`outputs/` 或 `logs/`
- **THEN** 常规架构边界测试 MUST 不因 ignored 文件存在而失败
- **AND** 若这些路径被 git 跟踪，测试 MUST 失败

### Requirement: 复杂度瘦身回流检查
项目健康护栏 MUST 能发现源码表面积中重新引入的低价值 package barrel、兼容 facade、重复 helper 聚合和 tracked runtime artifact。检查 MUST 只读取 tracked source、pyproject、README、docs、OpenSpec 和测试文件，不得扫描真实 `dataset/`、ignored `outputs/`、`logs/`、cache、checkpoint 或未跟踪本地 bytecode。

#### Scenario: 重依赖 barrel 回流被拒绝
- **WHEN** 已跟踪源码新增或扩大 package `__init__.py`，并 eager re-export 会导入 dataset、model、diagnostics、checkpoint registry、torch、pandas、matplotlib 或其它重依赖模块的符号
- **THEN** 架构边界检查 MUST 失败或要求该 re-export 有 current public 契约和轻量导入验证
- **AND** 失败信息 MUST 指向 owner module 直接导入或延迟导入收缩

#### Scenario: 兼容 facade 回流被拒绝
- **WHEN** 已删除的 builder、transform、BeamBench 聚合 owner、旧脚本 thin alias 或退役研究线 facade 重新出现在 tracked source 中
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求删除 facade 或在 active OpenSpec change 中登记 current public 契约

#### Scenario: 重复 helper 聚合需要理由
- **WHEN** 新增跨领域 helper 模块只收纳 CSV、JSON、float、slug 或 path 小工具，且没有明确 owner 和两个以上 current 调用点
- **THEN** 健康检查 MUST 失败或要求在 inventory 中登记为 `merge-candidate`、说明 owner 和验证命令
- **AND** 检查 MUST 不要求把领域私有 helper 强行移入全局 `utils`

#### Scenario: tracked runtime artifact 继续拒绝
- **WHEN** `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或非允许权重文件被纳入 git tracked 文件
- **THEN** 源码表面积检查 MUST 失败
- **AND** 未跟踪或 ignored 的同类本地产物 MUST 不驱动常规架构边界测试失败

### Requirement: 源码热点模块必须按职责收敛
项目 MUST 将继续增长的大文件拆分到职责明确的窄模块中。拆分后，公开入口 MAY 保留薄 facade 或兼容导出，但主要实现 MUST 位于按职责命名的模块中，不得重新形成新的私有聚合层。已退役的互补性分析模块 MUST 不再作为源码热点分层要求。

#### Scenario: 修改诊断 manifest 过滤逻辑不触碰 asset 写出
- **WHEN** 开发者调整当前诊断 manifest 的 scene、split、sample limit 或低质量样本过滤逻辑
- **THEN** 主要变更 MUST 位于对应当前 diagnostics owner 的过滤、cache 或 IO 相关模块
- **AND** 不需要修改 processed asset 写出、prediction summary 合并或 JEPA visual analysis 图表实现

#### Scenario: 修改 CSI hardening 不触碰 tokenizer
- **WHEN** 开发者调整 CSI hardening、pilot estimation 或噪声诊断逻辑
- **THEN** 主要变更 MUST 位于 CSI estimation 或 hardening 相关模块
- **AND** 不需要修改 CSI view tokenizer、view fusion 或 encoder registry glue

### Requirement: 热点模块 inventory 与回流防护
项目 MUST 维护热点模块拆分 inventory 或测试 allowlist，记录哪些模块仍作为兼容 facade 保留，哪些内部路径不得新增引用。架构边界测试 MUST 覆盖这些禁止回流路径。

#### Scenario: 架构测试拒绝内部 facade 回流
- **WHEN** 内部源码新增对已标记为兼容 facade 的二级聚合模块依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向推荐的窄模块路径

#### Scenario: inventory 更新
- **WHEN** 新增或拆分 scripts、tools、diagnostics helper 或大型 domain helper
- **THEN** 项目表面积 inventory 或等价文档 MUST 记录该入口的 lifecycle 和职责
- **AND** 测试 allowlist MUST 与文档保持一致

### Requirement: 热点拆分 inventory 必须覆盖优先级和禁止回流路径
项目 MUST 在表面积 inventory 或等价文档中记录热点模块拆分优先级、兼容 facade、推荐窄模块和禁止内部回流路径。架构边界测试 MUST 与 inventory 保持一致，并 MUST 对第一批 facade 执行行数上限、禁止片段和 helper 所属模块断言。

#### Scenario: inventory 记录第一批与第二梯队热点
- **WHEN** 开发者运行架构边界测试或审阅 `docs/project_surface_inventory.md`
- **THEN** inventory MUST 记录 `data/mmw/preparation.py`、trainer、dataset、run index、batch、diagnostics benchmark owner 和 evaluation pass 等当前热点的拆分方向
- **AND** inventory MUST 明确 HiST-Beam/Hist、viewer manifest 和 BGAM 专用 engine/model/evaluation 源码已退役，不作为当前热点清单成员
- **AND** inventory MUST 说明第二梯队热点的后续拆分方向或暂缓原因

#### Scenario: 内部代码不得从第一批 facade 回流导入 helper
- **WHEN** 内部源码新增对第一批 facade 中已迁移 helper 的 import 或调用
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应窄模块作为修复路径

### Requirement: 分层验证必须覆盖热点拆分
热点拆分实现完成后，项目 MUST 分层验证 OpenSpec、架构边界、focused 行为兼容和公开入口。所有项目相关 Python 验证 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 快速架构与 OpenSpec 校验
- **WHEN** 热点拆分任务完成
- **THEN** 开发者 MUST 运行 `openspec validate modularize-hotspot-modules --strict`
- **AND** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`

#### Scenario: 领域 focused tests 校验
- **WHEN** MMW preparation、diagnostics benchmark owner 或其它当前热点拆分完成
- **THEN** 开发者 MUST 运行对应 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`
- **AND** 若拆分触碰公开 CLI 或诊断入口，开发者 MUST 运行对应 help smoke 或 import smoke

#### Scenario: 全量回归作为最终验收
- **WHEN** 第一批热点拆分和架构防护全部完成
- **THEN** 开发者 MUST 运行 `conda run -n kd_mm_beam pytest -q` 作为最终回归验收
- **AND** 若全量测试因环境或本地数据缺失无法完成，最终说明 MUST 明确列出未运行原因和已完成的替代 focused 验证

### Requirement: 内部冗余检查可精简但外部边界检查保留
项目 MAY 删除内部私有 helper 中重复的 `assert`、重复类型检查、重复空值保护和只重新包装同类异常的 `try/except`，但用户输入、配置/manifest、文件路径、split/label-space/metric comparability、no-future-leak、输出产物边界和测试 fixture 契约相关检查 MUST 保留清晰失败模式。

#### Scenario: 删除内部重复检查
- **WHEN** 一个私有 helper 只由同 owner 调用，且调用方已经验证输入形状、类型或必需字段
- **THEN** 实现 MAY 删除该 helper 内重复的断言或二次类型检查
- **AND** focused tests MUST 证明正常路径输出、schema 和指标语义没有改变

#### Scenario: 保留用户输入边界检查
- **WHEN** CLI、manifest、配置文件、路径解析、数据 split、label space 或 checkpoint provenance 来自用户输入或外部文件
- **THEN** 系统 MUST 继续在边界处拒绝无效输入或记录明确 warning
- **AND** 错误或 warning MUST 足以定位无效字段、路径或不可比较原因

#### Scenario: 保留实验安全边界
- **WHEN** 代码处理 temporal source index、difficulty replay metadata、输出目录、cache、checkpoint 或真实实验产物
- **THEN** 系统 MUST 继续保证 no-future-leak、deterministic replay 和 ignored runtime artifact 边界
- **AND** 合并或删检查 MUST 不允许训练输出、日志、cache 或 checkpoint 进入源码变更

### Requirement: Benchmark runner suite 模块化边界
JEPA GPS shortcut benchmark runner MUST 保持公共 CLI 和输出 schema 兼容，同时将 suite-specific normalization、metric row construction、aggregation 和 artifact planning 拆分到职责明确的窄 helper 或模块。新增 suite 不得继续无边界扩大单一 runner facade。

#### Scenario: predictive suite helper 可独立测试
- **WHEN** runner 支持 `predictive_jepa_robustness` suite
- **THEN** predictive condition normalization、predictive metric row construction 和 predictive regional aggregation MUST 位于可单独导入测试的 helper 或窄模块中
- **AND** existing `run_jepa_gps_shortcut_benchmark` facade MUST 继续返回兼容 result dict 和 output_files

#### Scenario: 拆分不改变输出 schema
- **WHEN** benchmark runner 内部 helper 被拆分
- **THEN** `metrics_by_condition.csv`、`robustness_summary.csv`、`shortcut_reliance_summary.csv`、predictive summary JSON/CSV 和 `benchmark_manifest.json` 的核心字段 MUST 保持兼容
- **AND** focused tests MUST 验证旧 manifest 和 predictive smoke manifest 的 output registration

### Requirement: Runner 热点预算和暂缓理由
若 implementation 阶段无法安全拆分 benchmark runner，项目 MUST 在 `docs/project_surface_inventory.md` 登记新的热点预算、拆分方向和暂缓原因。暂缓登记 MUST 不替代未来拆分，但 MUST 防止热点静默扩大。

#### Scenario: 拆分暂缓但 inventory 更新
- **WHEN** implementation 判断 `jepa_gps_shortcut_benchmark.py` 拆分风险超过本 change 范围
- **THEN** inventory MUST 记录当前规模、suite-specific 拆分方向、暂缓原因和后续优先级
- **AND** 架构边界测试 MUST 能防止该 runner 在未登记的情况下继续显著扩大

#### Scenario: 后续新增 suite 前先处理预算
- **WHEN** 后续 change 计划为 benchmark runner 新增 suite、analysis family 或 artifact family
- **THEN** 维护者 MUST 先确认 runner 已拆分到窄模块或 inventory 中有明确预算和拆分任务
- **AND** 新增 suite MUST 不复制已有 difficulty corruption、aggregation 或 writer 逻辑

### Requirement: Benchmark runner 内部模块化
JEPA GPS shortcut benchmark runner SHALL 将 manifest/schema、suite-specific perturbation normalization、metric aggregation、artifact writing 和 plotting 拆分到职责明确的内部模块。原 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` MUST 保留为公开 facade，并 MUST 不承载新增 suite-specific helper 实现。

#### Scenario: 公开 facade 保持兼容
- **WHEN** 现有代码从 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` 导入公开 runner 或 analysis bundle helper
- **THEN** 导入 MUST 继续成功
- **AND** CLI `kd-sensing-jepa-gps-shortcut-benchmark` MUST 继续调用同一公开语义

#### Scenario: suite helper 不回流 facade
- **WHEN** 新增或修改 Scenario C、Scenario D、CxD 或 Predictive JEPA helper
- **THEN** 主要实现 MUST 位于对应窄模块
- **AND** facade MUST 只做兼容导出、薄 orchestration 或向后兼容包装

### Requirement: Benchmark facade 只暴露公开 runner API
JEPA GPS shortcut benchmark facade MUST 只暴露 CLI、runner、manifest loading、公开常量和下游分析需要的稳定 API。Suite-specific helper、metric normalization helper、summary helper 或 underscore private helper MUST 留在职责明确的窄模块中，facade MUST 不把它们重新导出为事实公共 API。

#### Scenario: CLI 继续使用公开 facade
- **WHEN** 用户执行 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help` 或通过包内 CLI 启动 benchmark
- **THEN** CLI MUST 继续导入公开 facade runner/API
- **AND** CLI MUST 不依赖 facade 重新导出的 private helper

#### Scenario: 测试直接覆盖窄模块 helper
- **WHEN** 单元测试需要验证 GPS query advantage normalization、metric summary 或 suite-specific helper
- **THEN** 测试 MUST 从 helper 所在窄模块导入目标符号
- **AND** 测试 MUST 不通过 `jepa_gps_shortcut_benchmark._private_name` 访问 helper

#### Scenario: facade 超预算时失败
- **WHEN** benchmark facade 重新承载已迁出的 helper 实现、重新导出 private helper 或超过维护索引声明的 facade 预算
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求将实现移回窄模块或删除不需要的 facade 导出

### Requirement: Benchmark 内部布局合并保持契约
JEPA GPS shortcut benchmark MAY 将同 owner 的内部 helper 合并为更少 Python 模块，但 public facade、manifest schema、suite normalization、perturbation semantics、comparability metadata、metrics CSV、benchmark manifest、图表产物和 runner CLI 行为 MUST 保持兼容。合并 MUST 不改变 P-suite、Scenario C、Scenario D、CxD 或 predictive robustness 的指标口径。

#### Scenario: Facade 行为保持不变
- **WHEN** 内部 `jepa_benchmark_*` helper 文件被合并或删除
- **THEN** `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` MUST 继续暴露当前 public benchmark 符号
- **AND** `kd-sensing-jepa-gps-shortcut-benchmark --help` MUST 继续可用
- **AND** public facade MUST 不吸收 benchmark runner、Scenario C/D、plotting 或 artifact registry 的主体实现

#### Scenario: Manifest 和输出 schema 保持不变
- **WHEN** benchmark runner 使用合并后的内部模块读取同一个 manifest
- **THEN** manifest validation、normalized suite config、metrics_by_condition、robustness_summary、benchmark_manifest 和 warnings 字段 MUST 与合并前语义兼容
- **AND** 未修改输入 manifest、训练配置、checkpoint、split CSV 或真实数据

#### Scenario: Scenario D 和 CxD 合并保持指标语义
- **WHEN** Scenario D/CxD normalization、phase diagram、dominance、failure-mode decomposition 或 metric-row helper 合并到更少 owner 模块
- **THEN** `scenario_d_image_observability` 与 `scenario_c_x_d_image_observability` suites MUST 继续输出相同条件字段、seed、difficulty digest、sample_count、metric、clean delta 和 comparability metadata
- **AND** 图表生成失败时仍 MUST 写出 metrics/manifest 并记录 warning

#### Scenario: Runner helper 合并保持产物边界
- **WHEN** runner summary、metric source ingestion 或 runner manifest helper 合并回 benchmark runner owner
- **THEN** evaluation-only、train-then-evaluate 和 reuse-existing-runs 协议 MUST 继续写入 ignored output directory 或 manifest 指定目录
- **AND** runner MUST 继续记录命令、环境、manifest digest、git status 摘要、模型配置、checkpoint provenance、difficulty provenance 和输出文件清单

### Requirement: Benchmark 冗余检查精简边界
JEPA GPS shortcut benchmark MAY 删除内部聚合、排序、标量转换和 row 派生 helper 中重复的二次检查，但 manifest validation、model comparability、suite normalization、perturbation determinism、Scenario C no-future-leak、Scenario D replay metadata 和 output artifact planning 的边界检查 MUST 保留。

#### Scenario: 内部 row helper 精简
- **WHEN** metric row、phase diagram、dominance ratio 或 summary helper 只消费 runner 已标准化的 rows
- **THEN** 实现 MAY 直接依赖标准化字段，删除重复类型检查和同义异常包装
- **AND** benchmark focused tests MUST 继续覆盖 summary rows、CxD rows 和 predictive rows 的核心字段

#### Scenario: 边界检查仍拒绝不可比较输入
- **WHEN** manifest 中模型的 split、sample_count、label_space、metric_profile、normalization artifact、difficulty profile digest 或 checkpoint provenance 不一致
- **THEN** benchmark MUST 继续拒绝写入同一严格可比较汇总或标记为不可比较
- **AND** 报告或 manifest MUST 记录不一致字段

#### Scenario: 扰动安全检查仍可测试
- **WHEN** Scenario C、Scenario D、CxD 或 predictive robustness suite 在 synthetic batch 上运行
- **THEN** repeated run with same seed MUST 保持 deterministic
- **AND** target label、sample id、未声明扰动的 modality 和 no-future-leak 约束 MUST 保持不变

### Requirement: 高级配置二次瘦身必须有候选分类
项目 MUST 在删除仍保留的高级实体 YAML 前维护候选分类。每个候选配置 MUST 被归入可由 recipe 无损生成、可由 recipe 生成但存在显式差异、或需要作为人工样例继续保留三类之一。

#### Scenario: 生成配置瘦身候选清单
- **WHEN** 开发者准备收敛 `configs/fusion/`、`configs/csi/hardening_matrix/` 或其它高级实验配置矩阵
- **THEN** 清单 MUST 记录每个候选实体 YAML 的分类、保留或删除理由和对应 recipe/overlay 名称
- **AND** 未分类的实体 YAML MUST 不得被删除

#### Scenario: 有差异的实体配置先记录差异
- **WHEN** 某个实体 YAML 与候选 recipe 在模型、loss、training schedule、dataset 字段或 checkpoint 来源上存在差异
- **THEN** 该差异 MUST 先记录为允许差异、overlay option 或保留理由
- **AND** 不得把该实体 YAML 当作无损可生成配置直接删除

### Requirement: 高级实体配置删除必须先分类
删除或迁移高级实体 YAML 前，项目 MUST 维护候选分类。每个候选配置 MUST 被归入 canonical/root 保留、recipe 可无损生成、recipe 可生成但有显式差异、人工样例、debug/smoke、diagnostics manifest、历史归档或删除。未分类配置 MUST 不得删除。

#### Scenario: JEPA image GPS 配置矩阵分类
- **WHEN** 开发者准备收敛 `configs/fusion/experiments/jepa_image_gps/*.yaml`
- **THEN** 每个实体 YAML MUST 有分类、保留/删除理由和替代 recipe、overlay、manifest 或文档路径
- **AND** 删除后的 README、docs、tests、scripts 和 OpenSpec current specs MUST 不引用不存在的 current 配置路径

#### Scenario: diagnostics manifest 保留
- **WHEN** 某个 YAML 是手工维护的 diagnostics manifest 且包含 checkpoint 占位、suite 定义或比较矩阵
- **THEN** 本 change MUST 保留该实体 YAML 或提供等价 manifest generator
- **AND** 删除前 MUST 有 focused test 验证 manifest 解析和输出 schema
