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

