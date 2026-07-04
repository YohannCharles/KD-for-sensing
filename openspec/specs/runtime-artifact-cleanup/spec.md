# runtime-artifact-cleanup Specification

## Purpose
定义本地运行产物清理的只读扫描、保护边界、显式删除阶段和候选分类契约，确保 outputs/logs/cache 中的临时产物可被审计整理，同时不会误删真实数据、源码、配置、OpenSpec 或可复现实验输入。
## Requirements
### Requirement: 清理候选 manifest
系统 MUST 提供本地运行产物清理候选 manifest 生成能力。manifest 生成过程 MUST 只读扫描，不得删除、移动、压缩或重写任何本地数据、输出、日志、cache、checkpoint、源码、配置、文档或 OpenSpec artifact。manifest MUST 为 machine-readable JSON，并记录每个候选路径、产物类型、大小、最近修改时间、匹配规则、候选原因、风险等级和保护状态。

#### Scenario: 生成 dry-run manifest
- **WHEN** 用户运行清理候选扫描命令并指定 `outputs/`、`logs/` 或 cache 根目录
- **THEN** 系统 MUST 写出 JSON manifest
- **AND** 系统 MUST 不删除、不移动、不压缩、不重写任何被扫描路径
- **AND** manifest MUST 包含候选总大小、候选数量、扫描根、规则版本和生成时间

#### Scenario: manifest 记录候选原因
- **WHEN** 某个路径因为 `_debug`、`_plan_check`、Python bytecode、pytest cache、失败 run、stale run、重复 checkpoint 或语义不清输出目录而进入候选
- **THEN** manifest MUST 记录匹配规则和人类可读原因
- **AND** manifest MUST 记录该路径的产物类型和大小

### Requirement: 保护边界优先
清理系统 MUST 默认保护真实数据、源码和可复现实验输入。`dataset/`、`All_models/`、`src/`、`configs/`、`docs/`、`openspec/`、`tests/`、已跟踪文件、OpenSpec artifacts、配置文件、文档和未明确匹配的活跃实验产物 MUST 不进入可删除候选；如果它们命中候选规则，manifest MUST 将其标记为 protected 并记录保护原因。

#### Scenario: 已跟踪文件受保护
- **WHEN** 扫描路径命中某条清理候选规则但该路径被 git 跟踪
- **THEN** 系统 MUST 将该路径标记为 protected
- **AND** 删除阶段 MUST 拒绝删除该路径

#### Scenario: 数据集目录受保护
- **WHEN** 扫描根包含 `dataset/` 或其子路径
- **THEN** 系统 MUST 默认不把真实数据文件列入可删除候选
- **AND** manifest MAY 记录数据目录大小摘要，但 MUST 标记为 protected summary

### Requirement: 显式删除阶段
系统 MAY 提供基于 manifest 的删除阶段，但删除阶段 MUST 要求用户显式传入 manifest 和确认参数。删除阶段 MUST 重新验证每个候选仍未受保护、路径仍在允许根下、路径状态与 manifest 记录兼容，并 MUST 写出删除结果报告。

#### Scenario: 未确认时拒绝删除
- **WHEN** 用户调用删除阶段但未提供显式确认参数
- **THEN** 系统 MUST 拒绝执行删除
- **AND** 系统 MUST 提示先检查 manifest 并提供确认参数

#### Scenario: 删除前重新验证保护状态
- **WHEN** manifest 中的候选路径在删除前变为已跟踪文件、活跃 run 或受保护路径
- **THEN** 系统 MUST 跳过该路径
- **AND** 删除结果报告 MUST 记录跳过原因

### Requirement: 清理候选分类
清理系统 MUST 将候选路径分类，至少覆盖 Python bytecode、pytest cache、短生命周期 debug/plan-check 产物、语义不清历史输出目录、失败或 stale run、重复 checkpoint、日志目录和个人备份压缩包。每个分类 MUST 有稳定的规则 ID。

#### Scenario: Python 缓存候选
- **WHEN** 扫描发现 `__pycache__/`、`.pyc` 或 `.pytest_cache/`
- **THEN** 系统 MUST 将其列为低风险缓存候选
- **AND** manifest MUST 使用稳定规则 ID 标记该候选

#### Scenario: 语义不清输出目录候选
- **WHEN** 扫描发现 `outputs/other/` 下的历史 run
- **THEN** 系统 MUST 将其列为需要人工确认的语义不清输出候选
- **AND** manifest MUST 保留 run index 摘要、checkpoint 数量和总大小

### Requirement: 退役 Hist 输出候选分类
清理系统 MUST 能将用户明确退役的 Hist/HiST-Beam、P3、V8/V9 probe、image-only Hist、history-anchor Hist、debug、smoke 和 plan-check 输出识别为本地运行产物候选。每个候选 MUST 记录稳定规则 ID、匹配原因、风险等级、大小、mtime 和保护状态。

#### Scenario: Hist 输出进入候选
- **WHEN** 扫描发现 `outputs/hist_beam_loso`、`outputs/history_anchor_*`、`outputs/image_only_legal_*`、`outputs/p3_v8_*` 或 `outputs/v9_*`
- **THEN** manifest MUST 将其列为退役 Hist 输出候选或需要人工确认候选
- **AND** manifest MUST 记录这些目录与已退役 Hist 研究线的关系

#### Scenario: debug 和 plan-check 输出进入低风险候选
- **WHEN** 扫描发现 `outputs/_debug_*`、`outputs/*_plan_check*` 或短生命周期 smoke 输出
- **THEN** manifest MUST 将其列为低风险或中风险清理候选
- **AND** manifest MUST 记录候选是否包含 checkpoint、metrics 或 source config

### Requirement: Hist 字符串不得作为唯一删除条件
清理系统 MUST 不得仅因路径包含 `hist` 字符串就删除产物。候选规则 MUST 结合 workflow 名称、run metadata、目录语义、退役清单或用户明确规则，避免误删历史窗口 baseline 或当前主线诊断。

#### Scenario: GPS history-window baseline 需要复核
- **WHEN** 扫描发现 `gps_window_*hist2` 或其它仅表示历史窗口长度的目录
- **THEN** manifest MUST 不得仅凭 `hist2` 将其归为 HiST-Beam 删除候选
- **AND** 如需删除，候选原因 MUST 来自 stale、debug、duplicate、用户显式模式或其它非裸字符串规则

### Requirement: runtime output 整理 manifest
系统 MUST 提供只读 runtime output 整理 manifest 能力，用于为 `outputs/` 中的历史训练 run、评估 run、registry checkpoint、analysis、cache 和 legacy 目录生成 move/archive/protect/review plan。整理 manifest 生成过程 MUST 不删除、不移动、不压缩、不重写任何本地数据、输出、日志、cache、checkpoint、源码、配置、文档或 OpenSpec artifact。

#### Scenario: 生成整理 dry-run manifest
- **WHEN** 用户对 `outputs/` 运行整理 dry-run
- **THEN** 系统 MUST 写出 machine-readable manifest
- **AND** manifest MUST 记录每个候选的 source path、建议 target path、action、artifact type、size、mtime、匹配原因、风险等级、保护状态和是否需要人工复核
- **AND** 系统 MUST 不移动或删除任何候选路径

#### Scenario: 分类 legacy 输出目录
- **WHEN** 整理扫描发现根级训练 run、`outputs/31/`、根级 `outputs/best_checkpoints/` 或 `outputs/eval_*`
- **THEN** manifest MUST 将它们分类为 legacy root run、legacy numeric scene、legacy registry 或 legacy evaluation
- **AND** manifest MUST 给出 canonical target 或 archive target
- **AND** 无法可靠判断 scope 的候选 MUST 标记为 `review` 或 `protect`

#### Scenario: cache 默认受保护
- **WHEN** 整理扫描发现 `outputs/cache/`
- **THEN** manifest MUST 默认将 cache 分区标记为 protected summary
- **AND** manifest MUST 不建议把 cache 移入训练 run、evaluation、analysis 或 archive

### Requirement: runtime output 整理执行阶段
系统 MAY 提供基于整理 manifest 的执行阶段，但执行阶段 MUST 要求用户显式传入 manifest 和确认参数。执行前 MUST 重新验证每个候选仍未受保护、source 仍在允许根下、source 状态与 manifest 兼容、target 不冲突且不会覆盖已有产物。执行阶段 MUST 写出 execution report。

#### Scenario: 未确认时拒绝整理执行
- **WHEN** 用户调用整理执行阶段但未提供显式确认参数
- **THEN** 系统 MUST 拒绝执行
- **AND** 系统 MUST 提示先检查整理 manifest 并提供确认参数

#### Scenario: 目标冲突时跳过
- **WHEN** manifest 中某个候选的 target path 在执行前已经存在且未声明可安全合并
- **THEN** 系统 MUST 跳过该候选
- **AND** execution report MUST 记录冲突 target 和跳过原因

#### Scenario: 路径变化时跳过
- **WHEN** manifest 中某个候选在执行前 size、mtime、保护状态或 git tracked 状态发生变化
- **THEN** 系统 MUST 跳过该候选
- **AND** execution report MUST 记录状态变化原因

### Requirement: Cleanup 历史规则必须有安全用途
runtime cleanup 和 organize MUST 保留 dry-run manifest、保护边界、显式确认、路径重验证和 execution report。只服务历史研究线考古、且不影响安全删除或当前输出整理的细粒度 legacy 规则 MUST 删除或降为文档说明。

#### Scenario: 删除低价值历史输出规则
- **WHEN** 某条 cleanup rule 只匹配已退役研究线的旧命名，且不参与当前 dry-run 安全保护
- **THEN** 本 change MAY 删除该规则
- **AND** manifest MUST 继续保护 tracked files、dataset、source/config/docs/OpenSpec、active run、cache 和 checkpoint 高风险路径

#### Scenario: 保留必要 legacy archive 分类
- **WHEN** organize dry-run 扫描根级 legacy run、legacy numeric scene、legacy registry 或 legacy evaluation
- **THEN** manifest MUST 继续给出 protect/review/archive/move action
- **AND** 执行阶段 MUST 继续要求显式确认和状态重验证

### Requirement: Cleanup 不能替代源码瘦身
源码表面瘦身 MUST 不调用 runtime cleanup 删除本地产物。runtime cleanup 只在用户明确要求清理本地产物时运行，并必须产生 manifest。

#### Scenario: 源码 change 不运行删除阶段
- **WHEN** 实施本 change 的源码、测试、配置或文档删减
- **THEN** 实现 MUST 不调用 cleanup execution 删除 `outputs/`、`logs/`、cache、checkpoint 或数据
- **AND** 如需整理本地产物，必须作为单独用户确认流程执行

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

### Requirement: 源码瘦身不得清理本地产物
本 change 的源码、测试、配置和文档删减 MUST 与本地产物清理分离。实现 MUST 不删除、移动或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或 TensorBoard 产物。

#### Scenario: 源码改动保护本地产物
- **WHEN** 本 change 删除或合并源码、测试、配置、脚本或文档
- **THEN** git diff MUST 不包含 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 中的新删除或修改
- **AND** 若用户另行要求清理这些路径，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

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

### Requirement: Ignored bytecode 清理不得混同源码删减
项目 MAY 清理 ignored Python bytecode 和 cache 目录，但 MUST 将其作为工作区噪声清理，而不是源码行为变更。该清理 MUST 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或历史权重。

#### Scenario: 清理 pycache 和 pyc
- **WHEN** 工作区存在 `__pycache__` 或 `.pyc` 文件
- **THEN** 本 change MAY 删除这些 ignored 本地文件
- **AND** 最终说明 MUST 明确它们不属于源码变更

#### Scenario: 实验产物仍受保护
- **WHEN** 候选路径位于 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/`
- **THEN** 本 change MUST 不自动删除该路径
- **AND** 若用户另行要求删除，流程 MUST 使用 runtime cleanup manifest 或单独显式确认

### Requirement: 源码与实验产物边界
项目 MUST 明确源码、配置、文档、OpenSpec artifacts 与本地数据、训练日志、缓存和输出产物的边界。本地运行产物 MUST 保持在 `.gitignore` 覆盖范围内，文档 MUST 指明哪些目录是可复现输入、哪些目录是可删除生成物。用户明确要求退役并清理某条失败实验路线或数据集工作流时，系统 MAY 删除匹配的本地 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 和训练诊断产物，但 MUST 先生成可审计清单并限制在未纳入源码且属于目标路线的路径内。

#### Scenario: 本地产物不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令
- **THEN** 生成的 logs、outputs、cache、checkpoint 和 Python bytecode 产物 MUST 位于忽略规则覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

#### Scenario: 文档说明产物边界
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `dataset/`、`All_models/`、`outputs/`、`logs/` 和 cache 目录的角色
- **AND** 文档 MUST 指明哪些目录通常不应纳入源码变更
- **AND** 文档 MUST 明确用户未要求清理时，源码删除不应自动清理历史 `outputs/`

#### Scenario: 清理旧失败实验产物
- **WHEN** 用户明确要求删除已退役失败路线的输出日志和实验结果
- **THEN** 清理流程 MUST 先写出 machine-readable manifest，记录每个候选路径、匹配原因、产物类型和大小
- **AND** 清理流程 MUST NOT 删除 `dataset/`、`All_models/` 已跟踪权重、OpenSpec artifacts、源码文件或未匹配失败路线的活跃实验产物

#### Scenario: 清理退役数据集工作流
- **WHEN** 用户明确要求删除 Raymobtime s008 代码和数据集
- **THEN** 清理流程 MUST 先写出 machine-readable manifest，记录每个 Raymobtime s008 候选数据、cache、日志、checkpoint、诊断和输出路径
- **AND** 清理流程 MUST 只删除 manifest 中属于 Raymobtime s008 的允许路径
- **AND** 清理流程 MUST NOT 删除其它数据集、外部未知 data_root、`All_models/` 已跟踪权重、OpenSpec artifacts 或非 Raymobtime 活跃实验产物

### Requirement: 内置权重与本地产物边界明确
项目 MUST 明确区分内置复现权重和本地生成 checkpoint。已跟踪的 `All_models` 权重如果继续保留，MUST 被文档标记为内置复现输入；新训练、评估或诊断产生的 checkpoint 和缓存 MUST 继续被忽略。

#### Scenario: README 说明 All_models 策略
- **WHEN** 开发者阅读 README 或扩展指南
- **THEN** 文档 MUST 说明 `All_models` 中已跟踪权重的用途、加载路径和是否属于源码仓库的可复现输入
- **AND** 文档 MUST 说明新生成的 `.pth` checkpoint 不应进入源码变更

#### Scenario: 新生成 checkpoint 不进入版本控制
- **WHEN** 用户运行训练、评估、预处理或诊断命令并生成 `.pth`、cache 或输出文件
- **THEN** 这些文件 MUST 位于 `.gitignore` 覆盖的路径或文件模式内
- **AND** 项目文档 MUST 不要求提交这些本地产物

### Requirement: 架构优化不得触碰本地数据和产物
源码、配置和入口表面积优化 MUST 不移动、删除、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、下载压缩包或其它本地运行产物。相关检查 MUST 只验证源码控制范围内的文件和忽略规则。

#### Scenario: 实施源码拆分不清理产物
- **WHEN** 开发者实施本 change 中的源码拆分、配置瘦身或入口收敛任务
- **THEN** 变更 MUST 不包含对 `dataset/`、`outputs/`、`logs/` 中真实文件的删除、移动或压缩操作
- **AND** 架构检查 MUST 继续只拒绝已跟踪源码表面积中的本地产物污染

#### Scenario: 数据目录策略不随本变更改变
- **WHEN** 本 change 完成并归档
- **THEN** 默认数据目录、legacy data_root 兼容规则和用户显式 data_root 行为 MUST 保持不变
- **AND** 本 change MUST 不要求用户迁移本地数据才能继续运行既有配置

### Requirement: 本 change 不改变本地产物策略
源码、配置和入口优化完成后，本地产物策略 MUST 保持现状。工作流 MAY 继续生成 outputs、logs、cache 和 checkpoint，但本 change MUST 不要求清理、压缩、迁移或提交这些产物。

#### Scenario: 训练输出仍位于忽略路径
- **WHEN** 用户在本 change 后运行训练或评估并生成输出
- **THEN** 新的 logs、outputs、cache 和 checkpoint MUST 继续位于 `.gitignore` 覆盖路径或显式本地输出目录
- **AND** 文档 MUST 不要求将这些本地产物加入源码变更

#### Scenario: 不要求清理已有产物
- **WHEN** 开发者实施本 change 的任务
- **THEN** 任务验收 MUST 不包含删除、压缩或迁移既有 `dataset/`、`outputs/`、`logs/` 文件
- **AND** 测试和 OpenSpec 校验 MUST 能在不修改这些本地产物的情况下完成

### Requirement: 语义化本地输出目录
项目 MUST 避免新脚本或默认配置继续向语义不清的兜底目录写入实验产物。长期保留的诊断脚本、local/manual helper 和 CLI 默认输出目录 MUST 包含实验族、数据集或能力名称；`outputs/other/` MAY 作为历史清理候选被扫描，但 MUST 不再作为新实验脚本的默认输出根。

#### Scenario: MMW modal15 历史输出目录可识别
- **WHEN** cleanup 或 run index 扫描到 MMW modal15 历史输出
- **THEN** 产物 root SHOULD 包含 `mmw_sunny_modal15` 或等价实验族名称
- **AND** 当前源码 MUST 不要求保留 MMW modal15 shell wrapper

#### Scenario: outputs other 不作为新默认值
- **WHEN** 架构边界测试扫描长期保留脚本和配置
- **THEN** 测试 MUST 拒绝新增默认输出根为 `outputs/other`
- **AND** 已存在的历史 `outputs/other/` 本地产物 MUST 只通过清理 manifest 管理

### Requirement: 清理流程不跨越源码边界
项目 MUST 将本地运行产物清理限定在 `.gitignore` 覆盖的本地产物范围内。清理工具、文档和测试 MUST 明确禁止删除源码、配置、文档、OpenSpec artifacts、已跟踪文件、`dataset/` 真实数据和 `All_models/` 历史复现权重。

#### Scenario: 清理 manifest 不含源码删除动作
- **WHEN** 用户生成清理候选 manifest
- **THEN** manifest MUST NOT 将 `src/`、`tests/`、`configs/`、`docs/` 或 `openspec/` 下的已跟踪文件列为可删除候选
- **AND** 如果这些路径被扫描到，manifest MUST 标记为 protected

#### Scenario: 文档说明本地产物边界
- **WHEN** 开发者阅读项目表面积 inventory 或 README
- **THEN** 文档 MUST 说明清理流程先生成 manifest
- **AND** 文档 MUST 说明真正删除需要用户显式确认

### Requirement: 退役研究线不触发本地产物隐式迁移
源码删除和包结构整理 MUST 与本地产物清理解耦。删除 Hist 源码 MUST 不自动移动、压缩或删除 `outputs/`、`logs/`、cache 或 checkpoint；本地产物删除 MUST 通过 runtime cleanup manifest 和显式删除阶段完成。

#### Scenario: 源码删除不隐式清理 outputs
- **WHEN** 实施者删除 Hist 源码、配置和文档入口
- **THEN** 该源码变更 MUST 不在同一步骤中用 ad hoc 命令删除 `outputs/`
- **AND** 需要删除的运行产物 MUST 先出现在 cleanup manifest 中

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

### Requirement: 本地工具状态跟踪必须被拒绝
项目健康护栏 MUST 拒绝将本地工具运行状态纳入源码跟踪。检查 MUST 至少覆盖 `.codegraph/daemon.pid`、`.codegraph/*.sock`、`.codegraph/*.db*`、`.pytest_cache/`、`__pycache__/`、`.pyc`、`outputs/`、`logs/`、cache 和新生成 checkpoint。

#### Scenario: CodeGraph daemon pid 被 tracked
- **WHEN** `git ls-files` 返回 `.codegraph/daemon.pid`
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 说明该文件是本地 CodeGraph 运行状态，应移出 git 跟踪并由 `.codegraph/.gitignore` 覆盖

#### Scenario: ignored 本地状态存在但未 tracked
- **WHEN** 工作区存在 ignored CodeGraph pid/socket/db、pytest cache 或 Python bytecode
- **THEN** 常规架构检查 MUST 不因文件存在失败
- **AND** 检查 MUST 只拒绝这些路径被 git 跟踪

### Requirement: Runtime cleanup 拆分必须保留 dry-run 与确认删除边界
Runtime artifact cleanup 重构 MUST 拆分扫描规则、manifest 渲染、delete/apply 校验和 organize 计划，并保留默认 dry-run 与破坏性操作显式确认。

#### Scenario: 删除仍需 manifest 与确认
- **WHEN** user invokes cleanup delete mode
- **THEN** 命令 MUST 要求同时传入 `--delete`、`--manifest <path>` 和 `--confirm-delete`
- **AND** 实现 MUST 在删除任何候选前重新校验 tracked/protected/path 状态

