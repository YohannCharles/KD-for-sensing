## ADDED Requirements

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
