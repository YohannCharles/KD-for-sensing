## ADDED Requirements

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
