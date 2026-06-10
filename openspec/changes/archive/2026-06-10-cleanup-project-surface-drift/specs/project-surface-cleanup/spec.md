## ADDED Requirements

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
