# project-health-guardrails Specification

## Purpose
定义项目健康护栏的检查层级、维护性热点 inventory、共享 pytest bootstrap、配置生命周期扫描和 OpenSpec 文档质量规则，使日常改动能快速发现入口漂移、旧路线回流、占位规范和本地产物边界问题。
## Requirements
### Requirement: 分层项目健康检查
项目 MUST 提供可记录、可重复的分层健康检查 workflow，用于在不启动真实训练、不读取真实数据、不写入源码内产物的前提下验证 OpenSpec、架构边界、CLI 入口和配置加载核心路径。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: 快速健康检查覆盖架构和入口
- **WHEN** 开发者运行项目快速健康检查
- **THEN** 检查 MUST 至少覆盖 OpenSpec strict validate、架构边界测试、CLI help smoke 和配置加载 characterization
- **AND** Python 检查命令 MUST 使用 `conda run -n kd_mm_beam pytest ...`
- **AND** 检查 MUST 不启动真实训练、不读取 `dataset/` 真实数据、不写入 checkpoint 或训练输出

#### Scenario: 领域改动追加 focused tests
- **WHEN** 实现改动触碰训练、数据集、诊断、CLI、配置解析或模型 forward
- **THEN** tasks 或最终验证说明 MUST 列出对应 focused tests
- **AND** focused tests MUST 优先覆盖被修改 workflow 的公开契约，而不是只运行全量 pytest

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

### Requirement: 测试启动基础设施集中
项目 MUST 使用 shared pytest bootstrap 管理测试导入路径和通用 fixture。普通测试文件 MUST 不再复制 `ROOT/SRC/sys.path.insert` 启动片段；需要隔离 import 边界的子进程 probe MAY 显式控制 `sys.path`，但该例外 MUST 局限在对应 probe helper 内。

#### Scenario: 普通测试使用 shared bootstrap
- **WHEN** 新增普通测试文件需要导入 `kd_sensing`
- **THEN** 测试 MUST 依赖 shared pytest bootstrap 或 editable install
- **AND** 测试文件 MUST 不复制 `sys.path.insert(0, str(SRC))` 作为文件级启动逻辑

#### Scenario: import-boundary probe 保留显式路径控制
- **WHEN** 架构边界测试在子进程中验证轻量导入或重依赖隔离
- **THEN** probe helper MAY 在子进程代码中显式设置 `sys.path`
- **AND** 该路径控制 MUST 不被抽成会 eager import runtime 模块的全局 helper

### Requirement: 健康护栏不改变 runtime 语义
项目健康护栏 MUST 只检查源码、配置、文档、OpenSpec 和测试基础设施一致性，不得改变训练、评估、预处理、模型 forward、数据 split、beam label、checkpoint schema 或本地产物边界。

#### Scenario: 健康检查无副作用
- **WHEN** 用户运行健康检查命令
- **THEN** 命令 MUST 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或真实本地运行产物
- **AND** 任何临时验证产物 MUST 位于 pytest 临时目录或 `.gitignore` 覆盖范围内

#### Scenario: 护栏实现不扩大公开入口
- **WHEN** 本 change 实现项目健康护栏
- **THEN** 系统 MUST 不新增长期训练/评估 CLI 或兼容 wrapper
- **AND** 若新增开发检查 helper，helper MUST 不成为 README 推荐的训练入口或旧研究路线替代入口

### Requirement: OpenSpec lifecycle inventory 完整性检查
项目健康护栏 MUST 检查 OpenSpec lifecycle inventory 覆盖 `openspec/specs/` 下的每个 capability。检查 MUST 在不读取真实数据、不启动训练、不写入本地产物的情况下运行，并 MUST 对未分类、重复分类或未知 lifecycle 值给出明确失败信息。

#### Scenario: lifecycle inventory 漏掉 spec
- **WHEN** 新增 `openspec/specs/<capability>/spec.md` 但未更新 lifecycle inventory
- **THEN** 架构边界或健康检查 MUST 失败
- **AND** 失败信息 MUST 指向新增 lifecycle 分类、确认 supporting/retired 状态或删除误建 spec 这几种修复路径

#### Scenario: lifecycle 值非法
- **WHEN** lifecycle inventory 使用不在允许集合中的分类值
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 列出允许值 `current`、`supporting` 和 `retired-tombstone`

### Requirement: 退役墓碑 wording 检查
项目健康护栏 MUST 检查 lifecycle 为 `retired-tombstone` 的 spec 是否明确包含退役、拒绝、历史或 migration guard 语义，并 MUST 拒绝未加退役限定的当前推荐入口、active mainline、默认 workflow 或可运行训练路线 wording。

#### Scenario: 墓碑 spec 缺少退役语义
- **WHEN** 某个 `retired-tombstone` spec 的 Purpose 和首个 requirement 都没有明确退役、不再支持或 migration guard 语义
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求补充退役墓碑说明或重新分类为 current/supporting

#### Scenario: 墓碑 spec 出现 active wording
- **WHEN** `retired-tombstone` spec 在未加历史/退役限定的段落中出现当前推荐入口、active mainline 或默认 workflow wording
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 指向对应 spec 和行号

### Requirement: 当前规格旧 active wording 漂移检查
项目健康护栏 MUST 检查 current specs、README、`docs/agent_navigation.md` 和 `docs/project_surface_inventory.md` 不得把已退役路线描述为当前推荐入口、active mainline、长期 orchestration 或必须实现的当前热点。已退役路线至少包括 HiST/Hist、Raymobtime s008、Top8 selector standalone workflow、GPS residual、camera residual、CRAF/MARF/G2D、Multimodal-NF、旧 KD 和旧静态 visualization workflow。

#### Scenario: project-architecture 拒绝未加退役限定的 Hist active wording
- **WHEN** `openspec/specs/project-architecture/spec.md` 出现未加退役限定的 HiST/Hist active mainline 或当前推荐入口描述
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 要求改为 retired-tombstone/supporting 语义或更新 lifecycle inventory

#### Scenario: README 或 inventory 恢复旧入口 wording
- **WHEN** README、docs workflow 或 project surface inventory 把退役路线写成 quickstart、当前推荐命令或长期入口
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求加入退役/历史限定或删除该推荐入口

### Requirement: 本地缓存和未跟踪产物不驱动健康检查失败
项目健康护栏 MUST 继续把 tracked source boundary 作为检查对象，不得因为开发者本地存在 ignored `__pycache__`、`.pytest_cache`、`outputs/`、`logs/` 或未跟踪实验产物而失败。若需要提示本地状态噪声，文档 MUST 通过 agent navigation 或最终说明解释，而不是让 CI 依赖本地 git ignored 状态。

#### Scenario: ignored cache 存在
- **WHEN** 工作树中存在 ignored Python bytecode、pytest cache 或本地 outputs
- **THEN** 常规架构边界测试 MUST 不因 ignored 文件本身失败
- **AND** 测试 MUST 继续拒绝这些产物被 git 跟踪

#### Scenario: tracked artifact 仍被拒绝
- **WHEN** `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/` 或非允许 checkpoint 被纳入 git tracked 文件
- **THEN** 源码表面积边界检查 MUST 失败
- **AND** 失败信息 MUST 指出这些路径属于本地运行产物或禁止跟踪产物

### Requirement: 主线实验文档索引检查
项目健康护栏 MUST 检查主线实验文档治理所需的 current 文档被创建、索引并登记生命周期。检查 MUST 不读取真实 `dataset/`、`outputs/`、checkpoint、cache、metrics 或 logs。

#### Scenario: 主线文档缺失或未索引
- **WHEN** 架构边界或文档健康检查运行
- **THEN** 检查 MUST 验证 README 或文档索引能指向主线模型目录、实验协议表和结果/claim 账本
- **AND** 缺失链接时检查 MUST 失败或给出明确修复信息

#### Scenario: 新增 capability 未登记 lifecycle
- **WHEN** `openspec/specs/mainline-experiment-documentation/spec.md` 存在
- **THEN** `docs/project_surface_inventory.md` 的 OpenSpec capability lifecycle 分类 MUST 将 `mainline-experiment-documentation` 标记为 current
- **AND** 未分类或分类为 retired/supporting 时检查 MUST 失败

### Requirement: current 文档结果状态 wording 检查
项目健康护栏 MUST 检查 current docs 和 current specs 不得把 mock、smoke、debug、lowmem、upper-bound、historical ablation、local substitute 或 blocked official reproduction 写成无 caveat 的正式结果。检查 MAY 使用关键词和限定词的静态规则，但 MUST 允许明确标记的历史或 appendix 段落。

#### Scenario: upper-bound 缺少限定
- **WHEN** current 文档中出现 `test_as_validation`、`upper-bound` 或等价用 test split 选 checkpoint 的说明
- **THEN** 附近文本 MUST 标明 upper-bound、非 official、不可作为 strict/unseen evaluation 或仅用于上限诊断
- **AND** 缺少限定时健康检查 MUST 失败

#### Scenario: future target 缺少历史限定
- **WHEN** current 文档中出现 `target-beam-source future`、`target_beam_source: future` 或等价 future target Table III 说明
- **THEN** 附近文本 MUST 标明 historical ablation、sequence-prediction ablation 或不得作为 Table III strict setup
- **AND** 缺少限定时健康检查 MUST 失败

#### Scenario: mock 或 smoke 数值缺少 caveat
- **WHEN** current 文档中出现 mock、dry-run、smoke、synthetic 或极小样本指标
- **THEN** 附近文本 MUST 说明该数值只验证代码路径或 schema，不用于论文/正式结果比较
- **AND** 缺少 caveat 时健康检查 MUST 失败

### Requirement: current spec 内部旧 active wording 检查
项目健康护栏 MUST 检查 current specs 内部不得同时保留旧 active workflow wording 与当前退役/拒绝 wording。已退役路线至少包括 legacy KD、teacher/student KD runtime、`teacher_no_kd`、`student_no_kd`、`logits_kd`、`rkd`、HiST/Hist、Raymobtime s008、standalone Top8 selector、GPS residual 和 camera residual。

#### Scenario: experiment-workflow 不保留 KD active 构建要求
- **WHEN** 健康检查扫描 `openspec/specs/experiment-workflow/spec.md`
- **THEN** 未加退役、拒绝、历史或 migration guard 限定的 `KD/loss`、`kd_mode`、teacher/student 成对训练、`student_no_kd` 当前入口 wording MUST 被视为漂移
- **AND** 检查 MUST 要求通过 OpenSpec change 清理为 current `model.primary` 语义或 retired/supporting 语义

#### Scenario: supporting helper 不被误判为 current workflow
- **WHEN** current spec 提到 TopK、LOSO、artifact registry、metric helper 或 migration guard
- **THEN** 文档 MUST 指明其 supporting、helper、guard 或被当前 workflow 消费的边界
- **AND** 文档 MUST 不把旧 standalone workflow 恢复为当前推荐入口

### Requirement: 文档健康检查无运行副作用
主线文档和规格漂移检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact 和测试文件。检查 MUST 不启动真实训练、不读取真实数据、不扫描 ignored 运行产物、不写入 checkpoint 或结果。

#### Scenario: 检查不读取本地产物
- **WHEN** 开发者运行文档健康检查或架构边界测试
- **THEN** 检查 MUST 不打开 `dataset/` 真实文件、`outputs/` metrics、checkpoint、feature cache 或 TensorBoard event
- **AND** 检查 MUST 只基于已跟踪文档、配置和 OpenSpec artifact 判断文档漂移

#### Scenario: Python 检查使用项目环境
- **WHEN** 文档健康检查通过 Python 测试实现
- **THEN** 推荐命令 MUST 使用 `conda run -n kd_mm_beam pytest ...`
- **AND** 该测试 MUST 不要求真实 GPU、真实数据或训练产物可用

### Requirement: 模型架构扩展护栏
项目健康护栏 MUST 检查新增模型、baseline 配置和扩展文档是否遵循模型架构扩展契约。检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact 和测试，不得读取真实 `dataset/`、`outputs/`、checkpoint 或 cache。

#### Scenario: 未说明例外的新整模型注册被发现
- **WHEN** 已跟踪源码新增 `@MODELS.register(...)` 或等价整模型注册名
- **THEN** 架构边界测试 MUST 要求该注册名出现在当前 specs、active change artifact、inventory 或明确 allowlist 中
- **AND** 缺少 whole-model exception 说明时检查 MUST 失败

#### Scenario: 文档默认示例绕过 modular_sequence 被发现
- **WHEN** 扩展指南把直接整模型注册描述为普通 baseline 的首选路径
- **THEN** 文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改为 `modular_sequence` 或子组件注册默认示例

### Requirement: 新模型 metadata 护栏
项目健康护栏或 focused tests MUST 验证新增可训练模型和模块化子组件的 metadata 可审计。整模型例外 MUST 提供 `training_strategy_metadata()` 或等价 helper；模块化组件 metadata MUST 能被 `ModularSequenceModel` 聚合。

#### Scenario: 整模型例外缺少 metadata
- **WHEN** 新增 whole-model exception 被 registry 构建
- **THEN** focused test MUST 验证其 metadata 至少包含模型类型、启用模态、架构类别和 reliability metadata 消费状态
- **AND** 缺失关键字段时测试 MUST 失败

#### Scenario: 模块化组件 metadata 可聚合
- **WHEN** 新增 encoder、core 或 head 提供训练策略 metadata
- **THEN** focused test MUST 验证 `ModularSequenceModel.training_strategy_metadata()` 包含该组件信息
- **AND** run metadata helper MUST 不需要为每个组件新增专用分支

### Requirement: Batch 和 runtime 分支回流检查
项目健康护栏 MUST 防止新增 baseline 通过复制 batch preparation、forward task routing 或 validation loop 来绕开共享 runtime。模型改动如需新增输入契约，MUST 更新中心化模态或 batch metadata contract，并补充 focused tests。

#### Scenario: 新 baseline 不新增专用 forward 分支
- **WHEN** 新增普通 baseline 配置或组件
- **THEN** 实现 MUST 复用 `prepare_task_inputs`、`forward_task_model` 或现有共享 runtime
- **AND** 架构或 focused tests MUST 拒绝仅服务某个 baseline 的训练/验证 forward 分支回流

### Requirement: 归档后 current spec 治理检查
项目健康护栏 MUST 检查归档后进入 `openspec/specs/` 的 current capability 同时具备 lifecycle inventory 分类、非占位 Purpose 和对应文档 caveat。检查 MUST 不读取真实 `dataset/`、`outputs/`、checkpoint、cache 或 logs。

#### Scenario: 新 current spec 缺少 lifecycle 分类
- **WHEN** `openspec/specs/<capability>/spec.md` 存在且该 capability 不在 `docs/project_surface_inventory.md` 的 lifecycle inventory 中
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向补充 `current`、`supporting` 或 `retired-tombstone` 分类

#### Scenario: 归档生成的 Purpose 未清理
- **WHEN** current spec 的 `## Purpose` 为空、长度不足或包含 `TBD`
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向对应 spec 文件

### Requirement: current JEPA 合法语境不被旧路线 guard 误判
项目健康护栏 MUST 继续拒绝退役路线 active wording，但 MUST 允许 current JEPA specs 和 diagnostics 中对现有 GPS-query baseline compatibility、condition-id 禁用字段和 forbidden-field diagnostics 的合法描述。

#### Scenario: GPS-query compatibility wording 被允许
- **WHEN** current JEPA spec 描述 `GPS-query` 或 `gps_query_pool` 作为现有 baseline compatibility、对照模型或默认行为兼容性
- **THEN** retired-route wording guard MUST 不把该行判定为退役路线回流
- **AND** 文档 MUST 不把该 baseline 写成旧 KD、HiST、Top8 selector standalone、GPS residual 或 camera residual 路线

#### Scenario: forbidden condition 字段诊断被允许
- **WHEN** current source 或 spec 记录 `condition_id_consumed=false`、`blocked_condition_fields`、`forbidden_condition_fields`、`gps_condition` 或 `image_condition`
- **THEN** 健康护栏 MUST 将其解释为防止 condition-aware router 的诊断或安全边界
- **AND** 只有在同一上下文把这些字段描述为模型直接输入或当前 router 入口时才应失败

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

