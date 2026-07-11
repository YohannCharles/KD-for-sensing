## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: 索引覆盖机器可读治理表
**Reason**: pyproject、inventory、registries 和 tests 已拥有这些事实，索引镜像导致漂移。
**Migration**: CLI 读 pyproject，scripts/configs 读 inventory，模型/branch 读源码与 current specs。

#### Scenario: 治理表不再复制
- **WHEN** architecture test 需要 current surface
- **THEN** 测试 MUST 从原始权威读取
- **AND** 不再从 maintainer index 读取 allowlist

### Requirement: Entrypoint owner metadata
**Reason**: Entrypoint metadata 已由 pyproject 和 project surface inventory 拥有。
**Migration**: Index route 只指向这两个权威来源。

#### Scenario: Entrypoint 事实单一来源
- **WHEN** console script 变化
- **THEN** pyproject 与 inventory MUST 更新
- **AND** index 不维护逐入口副本

### Requirement: package CLI 索引双向同步
**Reason**: 双向同步本身制造第三份 CLI 数据库。
**Migration**: pyproject 是命令事实，inventory 是 lifecycle 解释。

#### Scenario: CLI 删除不改 index 表
- **WHEN** console script 删除
- **THEN** implementation MUST 更新 pyproject、inventory 和 tests
- **AND** 不存在 index CLI row

### Requirement: Hotspot budget 行动元数据
**Reason**: 大量预算、状态和行动字段已从当前最小索引实现中移除。
**Migration**: 有真实行动的 hotspot 记录在 active change、inventory 或 focused test。

#### Scenario: Hotspot 不由 index 预算
- **WHEN** owner 需要拆分或保留
- **THEN** 决策 MUST 记录在对应 change/inventory
- **AND** index 不保存 budget entry

### Requirement: Hotspot metadata 不替代 inventory 解释
**Reason**: Index 不再保存 hotspot metadata，因此无需维护镜像一致性规则。
**Migration**: Inventory/change 独立承担解释。

#### Scenario: Hotspot 解释只有一个 owner
- **WHEN** 维护者查阅 hotspot 理由
- **THEN** 应读取 inventory 或 active change
- **AND** 不读取 index hotspot 摘要

### Requirement: Hotspot remediation wave metadata
**Reason**: Remediation wave 属于 change tasks/design，而不是长期导航索引。
**Migration**: 每个 architecture change 自己记录 wave、validation 和 rollback。

#### Scenario: Wave 从 change 读取
- **WHEN** agent 实施 architecture wave
- **THEN** MUST 读取 active change artifacts
- **AND** index 不保存 wave 副本

### Requirement: 维护索引必须记录架构尺寸基线和统计口径
**Reason**: 统计值易漂移且 inventory 已记录审计口径。
**Migration**: 需要时用 tracked-only 命令重算，并在 change/inventory 记录。

#### Scenario: 架构统计按需生成
- **WHEN** 维护者需要行数或文件数
- **THEN** MUST 从当前 tracked tree 重算
- **AND** index 不保存快照

### Requirement: 热点条目必须声明行动、验证和回滚信息
**Reason**: Index 不再拥有 hotspot entries。
**Migration**: 行动、验证和回滚写入 active change design/tasks。

#### Scenario: 热点任务可追踪
- **WHEN** hotspot 被纳入实现
- **THEN** active change MUST 记录其动作和验证
- **AND** index 只负责路由到该权威

### Requirement: remediation wave 必须可分阶段实施和回滚
**Reason**: 与 active change design/tasks 职责重复。
**Migration**: 使用 OpenSpec change artifact 记录分波和回滚。

#### Scenario: Wave 回滚不依赖 index
- **WHEN** wave 验证失败
- **THEN** implementation MUST 按 change design 回滚
- **AND** 不查询 index wave table

### Requirement: 维护索引必须覆盖新增二级热点
**Reason**: “所有热点必须入索引”导致索引膨胀为源码镜像。
**Migration**: 只有当前任务路由保留；热点按 change/inventory 管理。

#### Scenario: 新热点不扩大 index
- **WHEN** 审计发现大型 owner
- **THEN** 维护者 MUST 在对应 change/inventory 判断行动
- **AND** 不要求新增 index entry

### Requirement: 维护索引记录本次支持面收敛结果
**Reason**: 本 change 的结果由 pyproject、inventory、current specs 和 git diff 表达。
**Migration**: Index 只在 route 本身变化时更新。

#### Scenario: 删除结果不镜像
- **WHEN** 本 change 删除 CLI、scripts 或 source
- **THEN** implementation MUST 更新各自权威来源
- **AND** index 不复制删除 ledger

### Requirement: 维护索引统计基线必须声明口径
**Reason**: Index 不再记录统计基线。
**Migration**: `docs/project_surface_inventory.md` 或 active change 记录临时统计口径。

#### Scenario: 统计口径从 inventory 查询
- **WHEN** 维护者比较代码体量
- **THEN** MUST 使用 inventory/change 声明的 tracked-only 口径
- **AND** index 不保存数量
