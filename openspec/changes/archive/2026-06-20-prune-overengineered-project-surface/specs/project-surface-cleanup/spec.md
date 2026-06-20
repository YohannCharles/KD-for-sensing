## ADDED Requirements

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
