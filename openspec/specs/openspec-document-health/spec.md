# openspec-document-health Specification

## Purpose
定义 OpenSpec current specs、lifecycle inventory、Purpose hygiene、退役 wording 和文档健康检查边界，防止归档脚手架、未分类 capability 或旧 active wording 漂移进入当前规范。
## Requirements
### Requirement: 当前支持面漂移必须收敛
项目 MUST 维护当前支持面的脚本、配置、文档、测试和 OpenSpec 声明之间的一致性。发现当前入口引用不存在文件、inventory 统计与真实仓库不一致、或当前 spec 留有脚手架占位时，本次清理 MUST 修复漂移，而不是只放宽测试阈值。

#### Scenario: 修复已知支持面红点
- **WHEN** 开发者运行架构边界检查
- **THEN** 检查 MUST 不再因为 `configs/fusion/` 数量漂移、OpenSpec `TBD` Purpose 或已不存在的 hardening matrix 配置引用失败
- **AND** 对应文档和脚本 MUST 与真实仓库路径一致

#### Scenario: 更新 inventory 而非绕过约束
- **WHEN** 支持面清理需要改变保留配置、脚本或公开入口数量
- **THEN** `docs/project_surface_inventory.md` 和相关架构 guardrail MUST 同步更新
- **AND** 更新内容 MUST 解释保留、迁移或删除的类别边界

### Requirement: Root 文档支持面分类
项目 MUST 对仓库根目录和 `docs/` 中的长期文档、复现报告、研究笔记和历史方案进行生命周期分类。当前 README MUST 保持快速上手和主 workflow；长期需求与架构约束 MUST 留在 OpenSpec；研究/复现文档 MUST 标明用途和产物边界。

#### Scenario: Root 文档有生命周期
- **WHEN** 开发者查看项目表面积 inventory
- **THEN** inventory MUST 分类说明 README、README_REPRODUCE、环境/数据/报告文档、研究笔记和历史方案文档的当前用途
- **AND** 未分类 root 文档 MUST 被架构边界测试发现或要求补充说明

#### Scenario: 文档不推荐退役入口
- **WHEN** README 或长期 docs 描述当前可运行 workflow
- **THEN** 文档 MUST 不把已退役 KD/HiST/Top8/residual/camera residual 路线描述为当前推荐入口
- **AND** 如需保留历史背景，文档 MUST 明确标记为历史或退役记录

### Requirement: 治理表面不得复制源码事实
项目 MUST 避免用长期 YAML、测试 helper 或文档表格完整镜像源码目录、公开入口和热点预算。治理信息 MUST 只保留当前维护决策需要的最小结构化事实；可由 pyproject、OpenSpec、AST/path 扫描或真实文件树推导的事实 MUST 优先直接验证。

#### Scenario: 删除重复 allowlist
- **WHEN** 某个 allowlist 与 `pyproject.toml`、真实文件路径、OpenSpec lifecycle 或 README 当前入口重复表达同一事实
- **THEN** 本 change MUST 删除重复来源或把它降为说明性文档
- **AND** 健康检查 MUST 直接验证权威来源，而不是要求同步多份镜像表

#### Scenario: 保留必要防回流事实
- **WHEN** 某个退役 token、禁止入口、轻量导入边界或本地产物边界无法从代码自动推导
- **THEN** 项目 MAY 在小型 inventory、OpenSpec requirement 或测试常量中保留该事实
- **AND** 保留项 MUST 有明确用途，不得要求维护完整源码目录清单

### Requirement: 清理结果必须同步 current surface 文档
项目 MUST 在清理旧实验表面时同步 `docs/project_surface_inventory.md`、相关 README/docs、OpenSpec lifecycle 和架构边界测试。删除、降级或保留的候选项 MUST 有 owner、替代路径、验证命令和回滚方式。

#### Scenario: 删除后引用一致
- **WHEN** 本 change 删除或降级源码、脚本或配置
- **THEN** README、docs、tests 和当前 OpenSpec specs MUST 不再把旧路径声明为 current 支持入口
- **AND** 若保留历史说明，文档 MUST 明确标记为历史、local/manual 或兼容 reader

#### Scenario: 保留项有删除触发条件
- **WHEN** 某个 local/manual 脚本或 overlay YAML 因本地实验仍可能运行而保留
- **THEN** inventory 或实现说明 MUST 记录保留理由、输出边界和未来删除触发条件
- **AND** 保留项 MUST 不新增兼容 wrapper 或通用抽象层

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

### Requirement: 文档 wording 检查必须避免误伤合法当前语境
retired-route wording guard MUST 只拒绝未加退役/历史/拒绝限定的当前推荐入口表达。对于 current JEPA、diagnostics、runtime cleanup、legacy output classification 或 migration guard 的合法语境，测试 MUST 允许出现 retired token。

#### Scenario: 合法历史说明不失败
- **WHEN** README、docs 或 current spec 在退役、历史、migration guard、防回流或 archive 语境中提到 HiST、KD、BGAM、viewer manifest、Raymobtime、CRAF/MARF/G2D 或 Multimodal-NF
- **THEN** 健康检查 MUST 不失败
- **AND** 只有把这些路线写成当前推荐入口、默认 workflow 或 active mainline 时才 MUST 失败

#### Scenario: 结果 caveat 检查保留
- **WHEN** 文档出现 mock、smoke、upper-bound、local substitute 或 historical ablation 数值
- **THEN** 健康检查 MUST 继续要求附近文本包含 caveat
- **AND** 测试 MUST 不依赖固定长句，而应检查结构性限定词或结果状态字段

### Requirement: OpenSpec Purpose hygiene 必须被健康检查覆盖
项目健康护栏 MUST 扫描 current `openspec/specs/*/spec.md` 的 Purpose，拒绝 `TBD`、`created by archiving`、空 Purpose、未替换模板或其它归档脚手架文本。该检查 MUST 只读取已跟踪 OpenSpec artifact，不启动训练、不读取真实数据、不写入运行产物。

#### Scenario: current spec 保留归档 TBD
- **WHEN** current spec 的 Purpose 包含 `TBD - created by archiving`
- **THEN** 架构边界或 OpenSpec hygiene 检查 MUST 失败
- **AND** 失败信息 MUST 指向对应 spec 文件并要求补充真实 capability 边界说明

#### Scenario: OpenSpec validate 通过但 hygiene 失败
- **WHEN** `openspec validate --all --strict` 通过但项目自定义 hygiene 发现 scaffold Purpose
- **THEN** 项目 MUST 将其视为治理漂移
- **AND** 实施者 MUST 修复 Purpose 或在当前 change 中明确归档/折叠该 spec

### Requirement: Stale current 引用必须被健康护栏发现
项目健康护栏 MUST 检查 current README、docs、OpenSpec specs、inventory、tests 和 pyproject 中的当前入口引用是否指向真实存在的源码、配置或 console script。已删除入口不得继续作为 current scenario 或推荐命令出现。

#### Scenario: 已删除 BeamBench CLI 引用失败
- **WHEN** current spec 或 docs 要求 `kd_sensing.cli.beambench_check_dataset` 作为当前入口
- **THEN** 架构边界或 OpenSpec 校验任务 MUST 要求改为当前 owner module 或等价保留入口
- **AND** 该旧 CLI 文件 MUST 不因 spec 漂移被恢复

#### Scenario: 历史引用不误报
- **WHEN** archive、历史报告或明确标记为 retired/local/manual 的段落引用已删除入口
- **THEN** 健康护栏 MAY 允许该引用
- **AND** 该段落 MUST 不把旧入口描述为 quickstart、active mainline 或长期推荐 workflow

### Requirement: 文档与 OpenSpec 沉积必须可整理
README、docs 和 OpenSpec MUST 按职责维护当前行为，不得长期保留只描述历史迁移过程且不定义当前需求的正文。Archived spec 中的 TBD purpose MUST 被补齐或在后续归档整理中移除。

#### Scenario: README 保持入口导向
- **WHEN** 开发者阅读 README
- **THEN** README MUST 提供安装、环境、健康检查、主要入口和数据/产物边界
- **AND** 长实验矩阵、分析流程和当前诊断操作细节 MUST 通过 docs 或 OpenSpec 链接承载

#### Scenario: specs purpose 完整
- **WHEN** 开发者运行 OpenSpec 文档健康检查
- **THEN** 检查 MUST 拒绝新增 `TBD - created by archiving` purpose
- **AND** 既有 TBD purpose MUST 在本次整理范围内被替换为当前 capability 的真实目的说明

### Requirement: OpenSpec 文档健康检查结构化
项目 MUST 使用结构化方式检查 OpenSpec capability purpose。健康检查 MUST 只检查每个 spec 的 `## Purpose` 段落是否为空、过短或仍为归档占位文本，不得因为正文中描述被拒绝的占位文本而误判。所有项目相关 Python 检查 MUST 使用 `kd_mm_beam` 环境。

#### Scenario: purpose 检查不自引用误伤
- **WHEN** 某个 spec 正文描述健康检查会拒绝归档占位文本
- **THEN** 健康检查 MUST 不因正文出现该字符串而失败
- **AND** 检查 MUST 只根据 `## Purpose` 段落判断该 spec 是否存在文档健康问题

#### Scenario: purpose 问题报告具体 spec
- **WHEN** 某个 spec 的 `## Purpose` 段落为空、过短或仍为归档占位文本
- **THEN** 健康检查 MUST 报告该 spec 路径
- **AND** 报告 MUST 指向需要补齐的 capability purpose，而不是要求改写无关正文

### Requirement: OpenSpec 当前规范不得保留脚手架占位
当前 `openspec/specs/` 中的 spec MUST 具备真实 Purpose 和可理解的需求文本。归档 change 产生的 `TBD`、空泛占位或未替换模板文本 MUST 在进入当前规范后被修复，架构边界测试 MUST 能发现这类漂移。

#### Scenario: 当前 spec purpose 可读
- **WHEN** 开发者运行架构边界检查或 OpenSpec hygiene 检查
- **THEN** 当前 specs 的 Purpose MUST 是描述 capability 边界的真实文本
- **AND** Purpose MUST 不包含 `TBD`、未替换模板提示或归档脚手架说明

#### Scenario: 新归档规范进入当前面
- **WHEN** 一个 change 被归档并生成或修改 `openspec/specs/` 下的当前 spec
- **THEN** 归档后的 spec MUST 通过 OpenSpec 校验和项目架构 hygiene 检查
- **AND** 若归档工具留下占位 Purpose，开发者 MUST 在同一清理批次修复

### Requirement: Inventory 规模基线必须与真实仓库口径同步
项目表面积 inventory SHALL 记录当前源码、测试、脚本、配置和 OpenSpec 规模基线时说明统计口径，并在发现明显漂移时同步更新。规模数字 MUST 作为趋势和审计上下文，不得被解释为机械拆分、删除或放宽测试的唯一依据。

#### Scenario: 规模数字漂移被修复
- **WHEN** 维护者发现 inventory 中的 Python 文件数、配置数量、OpenSpec spec 数量或扫描日期明显落后于当前 tracked 文件系统
- **THEN** 本次文档健康修复 MUST 更新 inventory 的基线说明或改为更准确的可复核口径
- **AND** 更新 MUST 继续排除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和 ignored runtime artifacts

#### Scenario: 数字不替代右尺寸化判断
- **WHEN** inventory 记录某个源码、配置或 OpenSpec 数量
- **THEN** 文档 MUST 说明这些数字只是趋势信号
- **AND** 后续拆分、合并、保留或删除判断 MUST 继续依据 owner 职责、public surface、生命周期分类、调用边界和 focused validation

### Requirement: Agent context 文件纳入文档健康
项目文档健康检查 MUST 覆盖 agent context、atlas 或项目 skills 的引用一致性。新增 scoped context 或技能时，必须能从 AGENTS、agent navigation、inventory 或技能清单中定位其用途和适用范围。

#### Scenario: scoped context 引用失效
- **WHEN** scoped agent context 文件引用不存在的 spec、config、owner module 或验证命令
- **THEN** 文档健康或架构边界检查 MUST 失败
- **AND** 失败信息 MUST 指向失效引用

#### Scenario: 技能说明绕过 OpenSpec
- **WHEN** 项目级技能描述要求直接修改非平凡功能但不提 OpenSpec
- **THEN** 文档健康检查 MUST 要求补充 OpenSpec change 边界
- **AND** 技能 MUST 不把自己描述为需求契约权威

