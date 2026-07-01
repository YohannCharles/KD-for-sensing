# project-surface-cleanup Specification

## Purpose
定义项目源码表面、退役研究线和本地运行产物清理的长期边界，确保已退役 Hist/KD 入口不会以兼容 wrapper 或 virtual alias 回流，新的输出目录具备清晰语义，删除本地产物必须经过可审计 manifest。
## Requirements
### Requirement: 退役研究线源码表面清理
项目 MUST 支持按 OpenSpec change 退役整条研究线。退役后，该研究线的 CLI、配置、模型、engine、evaluation、测试和推荐文档入口 MUST 从当前支持面删除，且不得新增旧入口兼容 wrapper、virtual alias 或二级聚合层。

#### Scenario: Hist 研究线退役完成
- **WHEN** 开发者检查当前源码、配置、README、pyproject、tests 和 OpenSpec 当前 specs
- **THEN** 系统 MUST 不再声明 HiST-Beam/Hist CLI、`configs/hist_beam/`、`hist_beam_fusion` 或 Hist variants 为受支持入口
- **AND** 历史 archive MAY 保留旧记录，但 MUST 不作为当前支持契约

#### Scenario: 旧入口不被兼容接管
- **WHEN** 用户引用已退役的 Hist CLI、配置路径或模型注册名
- **THEN** 系统 MUST 失败或给出清晰退役错误
- **AND** 系统 MUST 不通过旧路径自动映射到其它当前 workflow

### Requirement: 输出目录用途分区
项目 MUST 为新的本地运行产物提供用途清晰的输出目录约定。训练 run、evaluation artifact、analysis artifact、cache、features、cleanup/organize manifest、scene/scenegroup best checkpoint 和 legacy archive MUST 采用可识别分区；当前支持 workflow MUST 不默认向语义不清的 `outputs/other/`、根级 `outputs/<run_name>/`、数字场景根 `outputs/31/` 或根级 `outputs/best_checkpoints/` 写入新产物。

Canonical 分区 MUST 至少包含：

- `outputs/cache/`: 可再生成 runtime cache。
- `outputs/cleanup_manifests/`: cleanup 或 organize dry-run/execution manifest。
- `outputs/analysis/`: 长期诊断、论文图、聚合分析和机器可读分析报告。
- `outputs/visual_analysis/`: JEPA visual analysis MAY 暂时保留为当前诊断出口；若迁移，MUST 迁入 `outputs/analysis/` 下的明确子目录。
- `outputs/evaluations/`: 评估集合和评估矩阵输出。
- `outputs/scene<id>/`: 单场景训练 run 和该 scene 的 best checkpoint registry。
- `outputs/scenegroup_<range-or-list>/`: 多场景训练 run 和该 scenegroup 的 best checkpoint registry。
- `outputs/archive/`: legacy root run、legacy `eval_*`、legacy numeric scene root 和需要保留但不作为当前入口的本地产物。

#### Scenario: 新训练输出写入可识别 scope
- **WHEN** 当前支持的训练 workflow 未显式覆盖输出目录
- **THEN** 默认输出 MUST 写入可识别的 scene 或 scenegroup 训练目录
- **AND** 运行目录 MUST 继续保存 `final_config.yaml`、`resolved_config.yaml`、metrics、checkpoint 和 runtime metadata
- **AND** 运行目录 MUST 不写入根级 `outputs/<run_name>/`、`outputs/31/<run_name>/` 或 `outputs/other/<run_name>/`

#### Scenario: 清理和整理 manifest 写入固定分区
- **WHEN** 用户运行 runtime cleanup 或 output organize dry-run
- **THEN** manifest MUST 写入 `outputs/cleanup_manifests/` 或用户显式指定的 manifest 路径
- **AND** manifest 路径 MUST 不与训练 run、evaluation、analysis、cache、features 或 registry checkpoint 混放

#### Scenario: legacy archive 不作为当前入口
- **WHEN** 本地存在迁移后的 `outputs/archive/` 产物
- **THEN** README、当前 docs、默认配置和 registry 解析 MUST 不把 archive 路径作为当前推荐入口
- **AND** 如需引用 archive 中的历史产物，文档 MUST 明确标记为历史记录或人工复核材料

### Requirement: 过时输出删除可审计
项目 MAY 删除用户明确要求退役的本地实验产物，但 MUST 先生成 machine-readable manifest，并且删除阶段 MUST 只处理未受保护、未被 git 跟踪、仍位于允许根内且匹配退役规则的候选。

#### Scenario: 删除退役 Hist 输出
- **WHEN** manifest 将 Hist/P3/V8/V9/debug/smoke/plan-check/stale 输出列为未受保护候选
- **THEN** 删除阶段 MAY 删除这些候选
- **AND** deletion report MUST 记录每个已删除、跳过或失败路径的原因

#### Scenario: 保护当前主线输出
- **WHEN** manifest 扫描到当前主线 analysis、features、cache、best checkpoint 或带 sidecar metadata 的复现 artifact
- **THEN** manifest MUST 默认将其标记为 protected 或需要人工确认
- **AND** 删除阶段 MUST 拒绝删除 protected 路径

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

### Requirement: Fusion 配置根目录必须保持可维护
`configs/fusion/` 根目录 MUST 只保留长期 canonical 配置或当前文档明确推荐的入口。实验特化、临时复现、低内存补丁、best/last 对照和一次性矩阵配置 MUST 迁移到明确实验子目录、归档说明或删除，且所有引用 MUST 同步更新。

#### Scenario: 收缩根目录 YAML
- **WHEN** 开发者检查 `configs/fusion/*.yaml`
- **THEN** 根目录 YAML 数量 MUST 回到架构 guardrail 允许范围内
- **AND** 每个保留 YAML MUST 能归入 canonical、当前推荐 workflow、或明确保留的薄入口配置

#### Scenario: 迁移配置后引用一致
- **WHEN** 某个 fusion YAML 被迁移、归档或删除
- **THEN** README、docs、scripts、tests 和 OpenSpec 当前 specs MUST 不再把旧路径声明为当前支持入口
- **AND** 若仍需保留复现路径，文档 MUST 指向新的明确位置或说明该配置已退役

### Requirement: 冗余源码删除必须保守
项目 MAY 删除无当前调用的源码 helper 或模块，但 MUST 先确认其不属于 console script、公开导出、注册入口、README/docs/OpenSpec 声明或测试依赖。无法确认外部依赖时，项目 MUST 优先保留源码并在 inventory 或后续 change 中记录待收敛项。

#### Scenario: 删除孤立 helper
- **WHEN** CodeGraph 或结构检查显示某个 helper 无当前内部调用
- **THEN** 开发者 MUST 进一步检查公开 API、配置注册、CLI、文档和测试引用
- **AND** 只有这些入口均不依赖该 helper 时，源码删除才可进入实现任务

#### Scenario: 保留可能的公共 API
- **WHEN** 某个无内部调用模块可能被外部脚本、公开导出或文档声明使用
- **THEN** 本 change MUST 不直接删除该模块
- **AND** 清理结果 MUST 记录保留原因或提出后续单独退役 change

### Requirement: 本地产物清理不得混同源码删除
项目 MUST 区分源码支持面清理和 ignored 本地产物清理。`__pycache__`、`.pyc`、`.pytest_cache`、egg-info 和明确临时备份 MAY 作为低风险本地清理项删除；`outputs/`、`logs/`、cache、checkpoint、dataset 和历史权重 MUST 不因源码清理自动删除。

#### Scenario: 删除低风险临时产物
- **WHEN** 用户要求清理冗余并且候选是 Python bytecode、pytest cache、egg-info 或明确临时备份
- **THEN** 清理 MAY 直接删除这些 ignored 本地产物
- **AND** 清理报告 MUST 明确它们不属于源码变更

#### Scenario: 保护实验输出
- **WHEN** 候选路径位于 `outputs/`、`logs/`、cache、checkpoint、`dataset/` 或历史权重目录
- **THEN** 本 change MUST 不自动删除该路径
- **AND** 若用户继续要求删除，流程 MUST 走 runtime cleanup manifest 或单独显式确认

### Requirement: 实验配置支持面分类
项目 MUST 对当前支持的配置文件进行生命周期分类，至少区分 canonical/root 推荐入口、实验复现配置、debug/smoke 配置、dataset preparation 配置、diagnostics 配置和已退役历史记录。新增、迁移或删除配置时，项目 MUST 同步更新 inventory、引用文档和相关架构 guardrail。

#### Scenario: 实验子目录配置有归属
- **WHEN** 开发者在 `configs/` 下新增实验特化 YAML
- **THEN** 该配置 MUST 位于语义明确的子目录或被 inventory 分类说明
- **AND** README、docs、OpenSpec 或脚本中的引用 MUST 指向真实存在的路径
- **AND** 该配置 MUST 不通过 root `configs/fusion/` 混入长期 canonical 入口，除非 inventory 明确将其列为当前推荐入口

#### Scenario: 配置引用漂移被发现
- **WHEN** 架构边界测试扫描 README、docs、scripts 和当前 OpenSpec specs 中的配置路径引用
- **THEN** 测试 MUST 能发现指向不存在配置文件的当前支持面引用
- **AND** 历史 archive 或明确标记为退役记录的引用 MUST 不被误判为当前入口

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

### Requirement: 审计确认的低价值源码表面必须收敛
项目 MUST 对已审计确认无当前调用方、无公开入口、无 registry、无 current 文档/OpenSpec 消费且仅由自身测试覆盖的源码表面执行删除或合并。删除 MUST 同步移除只服务该表面的测试、维护索引条目和 inventory current 分类；合并 MUST 不新增兼容 wrapper 或二级聚合层。

#### Scenario: 删除孤立诊断模块
- **WHEN** `communication_state_features` 或等价诊断 helper 只有自身测试引用，且不属于 CLI、配置、README、docs、OpenSpec current spec 或维护索引 current entry
- **THEN** 本 change MUST 删除该源码模块和只服务它的测试
- **AND** 架构边界检查 MUST 不再把该模块登记为当前诊断 surface

#### Scenario: 删除未接入模型原型
- **WHEN** LiDAR pillar encoder 或等价模型原型没有 registry、config、trainer、dataset、CLI 或 current docs 接入
- **THEN** 本 change MUST 删除该原型或将其移出当前源码支持面
- **AND** 当前 LiDAR BEV workflow MUST 保持可用且不要求该原型存在

#### Scenario: 合并重复 output registry helper
- **WHEN** 两个诊断 owner 提供等价的 `OutputRegistry` 或输出清单 helper
- **THEN** 本 change MUST 只保留一个 owner helper 或内联为局部函数
- **AND** 合并后 MUST 不新增长期通用 registry 抽象

#### Scenario: 删除未使用 dev 依赖
- **WHEN** dev extra 中的依赖没有源码、测试、docs、OpenSpec 或配置引用
- **THEN** 本 change MUST 从 `pyproject.toml` 删除该依赖
- **AND** 删除 MUST 不改变 runtime dependencies

#### Scenario: 源码删减不删除本地产物
- **WHEN** 本 change 删除源码、测试、配置或依赖声明
- **THEN** 实现 MUST 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重
- **AND** 若用户另行要求删除本地产物，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: Ponytail 二阶段源码表面瘦身
项目 MUST 将审计确认的过度工程表面按可验证 wave 收缩。候选项包括兼容 facade、legacy wrapper、单实现注册表、重复治理表、只服务已删表面的测试 helper、无收益样板 import 和可由现有标准库或既有依赖替代的默认依赖。每个候选项 MUST 被归类为删除、合并、保留并说明理由，且源码瘦身 MUST 不删除本地数据或运行产物。

#### Scenario: 删除默认重依赖
- **WHEN** 某个默认依赖只被当前源码用于标准图像读取、路径探测或其它可由已保留依赖覆盖的轻量任务
- **THEN** 本 change MUST 用更小的现有依赖或标准库替换该调用
- **AND** `pyproject.toml` MUST 不继续把该依赖列为默认 runtime 依赖

#### Scenario: 删除兼容 facade
- **WHEN** 某个 facade 只 re-export 已有 owner 模块符号，且 README、当前 docs、OpenSpec current specs、CLI、registry 和测试均可迁到 owner 路径
- **THEN** 本 change MAY 删除该 facade
- **AND** 内部源码 MUST 不新增对该 facade 的 import 来维持旧路径

#### Scenario: 折叠单实现扩展点
- **WHEN** 某个 registry、adapter 或策略接口只有一个 identity/no-op 实现且没有当前配置选择面
- **THEN** 本 change MAY 将其内联为默认路径或局部 helper
- **AND** 若未来出现第二个真实实现，项目 MUST 通过新的 OpenSpec change 重新引入窄扩展边界

#### Scenario: 样板 import 独立 wave 删除
- **WHEN** 项目 Python 版本契约已确认不低于 3.10 且代码不依赖 future annotations 的旧版本语义
- **THEN** 本 change MAY 批量删除 `from __future__ import annotations`
- **AND** 该机械修改 MUST 与行为修改分开验证或在最终说明中明确验证范围

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

### Requirement: 剩余低价值项目表面必须分类收敛
项目 MUST 对 ponytail 审计确认的剩余低价值表面建立候选分类，并按删除、合并、保留、归档或后续 change 处理。候选范围 MUST 至少覆盖重复实体 YAML、薄 re-export facade、退役 route guard、无调用 helper、一次性分析脚本、重复小工具和大型治理测试。分类 MUST 记录公开 surface 风险、当前调用方、替代 owner、验证命令和回滚方式。

#### Scenario: 候选删除项有证据
- **WHEN** 开发者准备删除源码、配置、脚本或测试
- **THEN** 候选项 MUST 被证明不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 删除计划 MUST 指向替代 owner、recipe、文档位置或说明无需替代

#### Scenario: 候选保留项有理由
- **WHEN** 某个候选项因 public API、人工样例、diagnostics manifest 或外部迁移风险被保留
- **THEN** inventory 或实现说明 MUST 记录保留理由和未来删除触发条件
- **AND** 项目 MUST 不为了保留该候选项新增兼容 wrapper 或第二套治理表

### Requirement: 一次性研究脚本不得长期占据 current surface
只服务已完成调试结论、历史 sweep 汇总或人工复盘的一次性脚本 MUST 从当前支持面删除、归档为历史文档，或明确标为 local/manual research artifact。保留脚本时 MUST 记录 owner、输入输出边界和仍需运行的 focused 验证；删除脚本时 MUST 保留必要结论和 caveat 到 docs 或报告。

#### Scenario: 删除 CSI hardening sweep analyzer
- **WHEN** `scripts/analyze_csi_hardening_sweep.py` 或等价一次性脚本只服务历史 CSI hardening 调试结论
- **THEN** 本 change MAY 删除该脚本和只服务它的测试
- **AND** 仍有价值的结论 MUST 留在 `docs/research_notes.md`、CSI hardening 文档或对应报告中

#### Scenario: 保留当前诊断脚本
- **WHEN** 某个 `scripts/` 文件仍被 README、docs、OpenSpec current spec 或 package workflow 明确引用
- **THEN** 本 change MUST 不删除该脚本
- **AND** 若脚本保留，inventory MUST 将其分类为 research diagnostic、dataset preparation、figure helper 或 manual/local script

### Requirement: 源码瘦身不得触碰本地产物
项目表面瘦身 MUST 只修改源码、配置、测试、文档和 OpenSpec artifact。实现 MUST 不删除、不移动、不压缩、不重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event、`All_models/` 历史权重或其它本地运行产物。

#### Scenario: 实现 wave 检查工作树
- **WHEN** 每个源码瘦身 wave 完成
- **THEN** 开发者 MUST 检查 `git status --short`
- **AND** 新增或修改内容 MUST 不包含本地数据、输出、日志、cache、checkpoint 或临时训练产物

#### Scenario: 用户另行要求清理输出
- **WHEN** 用户要求删除 `outputs/`、`logs/`、cache、checkpoint 或 dataset 内容
- **THEN** 该操作 MUST 走 runtime cleanup manifest 或单独显式确认
- **AND** 本 change 的源码瘦身任务 MUST 不把该清理混入同一删除 wave

### Requirement: Ponytail 审计候选必须有删除证据
项目 MUST 在删除或合并低价值源码、配置、脚本、测试或文档入口前记录最小证据：当前调用方、公开入口风险、替代 owner、是否被 registry/CLI/current docs/OpenSpec 消费、验证命令和回滚方式。没有证据的候选 MUST 保留或降为后续单独 change。

#### Scenario: 删除候选具备证据
- **WHEN** 开发者准备删除 ponytail 审计列出的候选项
- **THEN** change artifact 或实现说明 MUST 记录该候选不属于当前 package CLI、registry、canonical config、README/docs current 入口、OpenSpec current requirement 或必要 focused test 输入
- **AND** 记录 MUST 指向替代 owner、替代 recipe、普通 unknown-name 行为或说明无需替代

#### Scenario: 保留候选具备理由
- **WHEN** 某个候选因 public API、外部脚本风险、manifest 安全边界或当前 workflow 消费而保留
- **THEN** inventory、任务说明或最终实现说明 MUST 记录保留理由
- **AND** 项目 MUST 不为保留该候选新增兼容 wrapper、二级聚合层或重复治理表

### Requirement: 源码瘦身不得清理本地产物
本 change 的源码、测试、配置和文档删减 MUST 与本地产物清理分离。实现 MUST 不删除、移动或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或 TensorBoard 产物。

#### Scenario: 源码改动保护本地产物
- **WHEN** 本 change 删除或合并源码、测试、配置、脚本或文档
- **THEN** git diff MUST 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 中的新删除或修改
- **AND** 若用户另行要求清理这些路径，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: 退役保活测试必须删除或迁移
只用于证明退役类、旧 alias、旧 facade 或已删除 helper 仍可直接导入/forward 的测试 MUST 删除或改写为当前 owner、registry unknown-name、canonical config 或 CLI 行为测试。

#### Scenario: 退役模型 direct-forward 测试不再保活
- **WHEN** 已从 registry 退役的整模型类不再属于当前公开 API
- **THEN** 测试 MUST 不再直接实例化该类来证明其 forward 仍可用
- **AND** 相关覆盖 MUST 迁到当前 `modular_sequence`、feature extractor、registry unknown-name 或 config load 行为

### Requirement: Ponytail 审计表面必须分类收口
项目 MUST 将 ponytail 审计确认的临时配置、一次性脚本、root runbook、薄 facade、重复 helper 和本地工具状态分类为删除、迁移、保留或后续 change。分类 MUST 记录 owner、当前调用方、公开 surface 风险、替代入口、验证命令和回滚方式。未分类项 MUST 不得作为 current README、docs、OpenSpec 或 package CLI 推荐入口。

#### Scenario: 新增脚本被分类
- **WHEN** `scripts/` 或 `tools/analysis/` 下存在新增 Python/shell 脚本
- **THEN** inventory MUST 将其分类为 package_cli、research_diagnostic、dataset_preparation、figure_helper、shell_orchestration 或 local/manual artifact
- **AND** 未分类脚本 MUST 不得被 README、docs 或 OpenSpec 描述为当前推荐入口

#### Scenario: 临时配置不进入 root canonical surface
- **WHEN** 新增配置只服务 Scene31/RBMA queue/fullrun/strong-encoder/seed sweep 或其它本地实验编排
- **THEN** 配置 MUST 位于语义明确的 experiment 子目录、被归档说明，或被删除
- **AND** 配置 MUST 不直接混入 `configs/fusion/*.yaml` 根目录，除非 inventory 将其登记为 canonical/current thin entry

### Requirement: 内部代码不得通过公开 facade 回流导入 owner helper
公开 facade MAY 保留外部兼容 import 或 CLI glue，但内部源码 MUST 直接导入职责明确的 owner 模块。新增内部引用不得从 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark`、`kd_sensing.data.mmw.preparation` 或其它已登记 facade 导入已迁出的窄 helper，除非该文件本身就是 facade 或 package CLI glue。

#### Scenario: JEPA visual analysis 直连 benchmark owner
- **WHEN** `kd_sensing.diagnostics.jepa_visual_analysis` 需要 benchmark suite 常量或 analysis bundle reader
- **THEN** 它 MUST 从 `jepa_benchmark_common.py`、`jepa_benchmark_runner.py` 或对应窄 owner 导入
- **AND** 它 MUST NOT 通过 `kd_sensing.diagnostics.jepa_gps_shortcut_benchmark` facade 获取这些 helper

#### Scenario: CLI 仍可使用公开 facade
- **WHEN** package CLI 需要调用 GPS shortcut benchmark runner
- **THEN** CLI MAY 继续使用公开 facade 或直接 owner
- **AND** 该例外 MUST 不允许 diagnostics、engine、data、models 或 tests 内部新增 facade 回流

### Requirement: CodeGraph 本地运行状态不得作为源码产物
CodeGraph 索引、daemon pid、socket、WAL、cache、log 和 hook marker MUST 被视为本地运行状态。源码仓库 MAY 跟踪 `.codegraph/.gitignore` 这类忽略规则，但 MUST NOT 跟踪 `.codegraph/daemon.pid`、daemon socket、数据库、WAL、cache 或其它会随本机进程变化的文件。

#### Scenario: daemon pid 不被跟踪
- **WHEN** 开发者运行 `git ls-files .codegraph/daemon.pid`
- **THEN** 该路径 MUST 不在 git tracked 文件列表中
- **AND** `.codegraph/.gitignore` MUST 覆盖 pid、socket、数据库和 cache 等本地 CodeGraph 状态

#### Scenario: CodeGraph 仍可本地运行
- **WHEN** CodeGraph daemon 在本地生成 pid、socket 或数据库文件
- **THEN** 这些文件 MAY 保留在工作区供本地工具使用
- **AND** 架构边界检查 MUST 不要求提交这些文件

### Requirement: Fusion 根配置与 inventory 必须一致
`configs/fusion/` 根目录的实体 YAML MUST 与 `docs/project_surface_inventory.md` 中的 root canonical/current 分类一致。若 root YAML 集合变化，项目 MUST 同步更新 inventory、引用文档和架构边界测试；若文件不属于 root canonical/current thin entry，项目 MUST 将其迁入 experiment 子目录、归档或删除。

#### Scenario: root YAML 集合被验证
- **WHEN** 架构边界测试扫描 `configs/fusion/*.yaml`
- **THEN** 每个根 YAML MUST 出现在 inventory 的 root 保留分类中
- **AND** 未登记 root YAML MUST 被视为支持面漂移

#### Scenario: 实验 YAML 迁移后引用同步
- **WHEN** root fusion YAML 被迁移到 `configs/fusion/experiments/<family>/`
- **THEN** README、docs、scripts、tests 和 OpenSpec 中的 current 引用 MUST 指向新路径或移除
- **AND** 历史引用 MUST 明确标记为 historical、retired 或 local/manual

