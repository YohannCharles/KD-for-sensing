# maintainer-context-index Specification

## Purpose
定义中心化、机器可读的维护上下文索引，使 AI agent、维护者和架构边界测试能从稳定位置读取项目任务路由、治理事实、生命周期入口和无运行副作用边界，同时不替代 OpenSpec requirements、README quickstart、AGENTS 操作规则或项目表面积 inventory 的审计解释职责。
## Requirements
### Requirement: 中心化维护上下文索引
项目 SHALL 提供稳定路径的最小机器可读任务路由索引。索引 MUST 只记录无法从 pyproject、current specs、inventory 和源码路径直接推导的 route id、scoped context、authority paths、owner roots、focused validation 和 retired-route guard 引用；MUST 不保存 entrypoint、config、hotspot、wave 或源码统计镜像。

#### Scenario: 索引文件可定位
- **WHEN** agent 或维护者准备进行非平凡改动
- **THEN** 项目 MUST 提供 `docs/maintainer_context_index.yaml` 或等价稳定路径
- **AND** 索引 MUST 声明自己不是运行配置、entrypoint database、hotspot budget table 或源码树镜像

#### Scenario: 索引不成为运行时入口
- **WHEN** 用户运行训练、评估、预处理、诊断或 cleanup
- **THEN** runtime MUST 不读取维护上下文索引
- **AND** 本地产物状态 MUST 不影响索引读取

### Requirement: 索引覆盖 AI 任务路由
维护上下文索引 SHALL 记录模型、数据、配置、CLI、诊断、OpenSpec、文档、claim 和 atlas 路由。每条 route MUST 只包含 scoped context path、authority paths、owner roots、focused validation 和必要 caveat，不得复制 owner 文件清单或 capability requirements。

#### Scenario: 模型改动定位 scoped context
- **WHEN** agent 修改模型、forward 或 registry
- **THEN** 索引 MUST 指向模型 context、权威 specs、owner root 和 focused tests
- **AND** 详细设计理由 MUST 留在 OpenSpec/inventory

#### Scenario: CLI 或配置改动定位权威来源
- **WHEN** agent 修改 CLI、script 或 config
- **THEN** 索引 MUST 指向 pyproject、inventory、对应 scoped context 和验证命令
- **AND** 索引 MUST 不复制完整 CLI 或 config allowlist

### Requirement: 索引 schema 可轻量验证
项目 SHALL 仅验证索引的 route id 唯一性、context/authority/owner path 存在性、focused validation 格式和 retired-route guard 引用。验证 MUST 不要求治理镜像 section、entrypoint lifecycle table、hotspot budget 或统计基线。

#### Scenario: route 缺少必填字段
- **WHEN** route 缺少 id、context path、authority paths、owner roots 或 focused validation
- **THEN** 轻量检查 MUST 失败并指出 route
- **AND** 检查 MUST 不要求补回已删除镜像字段

#### Scenario: focused command 环境正确
- **WHEN** route 包含项目 Python 验证命令
- **THEN** 命令 MUST 使用 `conda run -n kd_mm_beam`
- **AND** OpenSpec target MUST 是 current spec、active change 或 `--all`

### Requirement: 索引与权威来源对齐
维护上下文索引 SHALL 与 AGENTS、AI 维护导航、project surface inventory、OpenSpec specs、pyproject 和源码文件存在性保持一致。索引 MAY 摘要这些来源的路径和分类，但 MUST 不覆盖 OpenSpec requirement 或 README quickstart 的当前推荐入口判断。

#### Scenario: 索引引用不存在文件
- **WHEN** 索引登记的源码、脚本、配置、文档或 OpenSpec spec 路径不存在
- **THEN** 架构边界或文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求删除误登记项、恢复文件或更新索引路径

#### Scenario: OpenSpec capability 未登记 lifecycle
- **WHEN** `openspec/specs/<capability>/spec.md` 存在但索引或 inventory 未提供 lifecycle 分类
- **THEN** 健康检查 MUST 失败
- **AND** 失败信息 MUST 要求补充 `current`、`supporting` 或 `retired-tombstone` 分类

### Requirement: 索引变更无运行副作用
维护上下文索引能力 SHALL 只影响文档、OpenSpec artifact、测试治理数据和静态健康检查。实现该能力 MUST 不改变训练、评估、预处理、模型 forward、数据 split、配置解析、checkpoint schema、输出目录或本地产物清理语义。

#### Scenario: 实现索引不改变 runtime
- **WHEN** 本 change 完成
- **THEN** 项目 MUST 不新增长期训练/评估/预处理 CLI
- **AND** 项目 MUST 不修改默认 dataset 读取、模型构建、metric 计算、checkpoint 写出或 runtime output 分区

#### Scenario: 索引验证不读取本地产物
- **WHEN** 开发者运行索引相关健康检查
- **THEN** 检查 MUST 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event
- **AND** 检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact、pyproject 和测试文件

### Requirement: 维护索引 validation 命令必须引用当前可运行目标
`docs/maintainer_context_index.yaml` 中记录的 focused validation 命令 MUST 引用当前可运行的 OpenSpec spec、active change 或通用 strict 校验。若命令使用 `openspec validate <name> --strict` 形式，`<name>` MUST 是当前存在的 spec 名称、active change 名称，或被明确标记为历史记录且不得出现在可复制执行的 focused validation 列表中。归档后无 active change 时，索引 MUST 优先记录 `openspec validate --all --strict` 或 current spec validation，而不是已归档 change 的 validation 命令。

#### Scenario: focused validation 不引用已归档 change
- **WHEN** `openspec list --json` 不包含某个 change，且 `docs/maintainer_context_index.yaml` 的 focused validation 列表包含 `openspec validate <change> --strict`
- **THEN** 架构边界或文档健康检查 MUST 失败
- **AND** 失败信息 MUST 要求改用 `openspec validate --all --strict`、current spec validation 或恢复/说明 active change 状态

#### Scenario: 无 active change 时使用全量 OpenSpec strict
- **WHEN** `openspec list --json` 返回空 active change 列表
- **THEN** 维护索引的 focused validation MUST 包含 `openspec validate --all --strict` 或等价 current specs strict 校验
- **AND** 维护索引 MUST 不把 archive change validation 当作当前可执行验收入口

### Requirement: 维护上下文索引可服务 agent routing
维护上下文索引 MUST 提供足以支持 agent routing 的最小结构化字段，例如 task route id、权威文档路径、owner module、focused validation 和 retired-route guard。索引 MUST NOT 扩展为完整源码目录镜像、完整 config 数据库或完整 OpenSpec requirement 复制。

#### Scenario: agent 查询任务路由
- **WHEN** agent 需要判断模型、配置、CLI、诊断、claim 或文档任务的先读内容
- **THEN** 维护上下文索引 MUST 提供对应 route 的最小事实
- **AND** 详细 rationale MUST 仍由 inventory、README 或 OpenSpec spec 提供

### Requirement: Atlas 输出引用权威来源
如果项目提供 spec/config/claim atlas，atlas MUST 引用权威路径、capability lifecycle、owner、验证命令和 caveat，并 MUST 标记生成时间或来源。Atlas MUST 不覆盖 current specs、inventory 或 claim registry。

#### Scenario: atlas 与 inventory 冲突
- **WHEN** atlas 和 inventory 对同一 capability lifecycle 给出冲突信息
- **THEN** agent MUST 将其视为治理漂移
- **AND** 后续变更 MUST 同步 atlas 生成源、inventory 和相关 specs

