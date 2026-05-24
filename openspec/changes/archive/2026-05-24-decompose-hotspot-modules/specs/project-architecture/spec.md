## ADDED Requirements

### Requirement: 热点模块拆分边界
项目 MUST 为高变更频率的大型模块提供职责拆分路径。拆分后的窄模块 MUST 按 schema/constants、pure helper、reader、writer、orchestration 或 domain-specific adapter 组织，公开 facade MAY 保留兼容导出，但新内部代码 MUST 优先依赖窄模块。

#### Scenario: 新内部代码使用窄模块
- **WHEN** 开发者在训练、评估、预处理、诊断或 viewer 相关实现中新增代码
- **THEN** 新代码 MUST 优先从职责明确的窄模块 import
- **AND** 不得新增对仅用于兼容 re-export 的二级聚合模块的内部依赖

#### Scenario: 公开入口兼容
- **WHEN** 现有用户从公开 facade import 旧符号
- **THEN** 导入 MUST 继续成功，除非对应 change 明确声明 breaking change
- **AND** facade MUST 不触发比旧路径更重的 eager import

### Requirement: 热点模块 inventory 与回流防护
项目 MUST 维护热点模块拆分 inventory 或测试 allowlist，记录哪些模块仍作为兼容 facade 保留，哪些内部路径不得新增引用。架构边界测试 MUST 覆盖这些禁止回流路径。

#### Scenario: 架构测试拒绝内部 facade 回流
- **WHEN** 内部源码新增对已标记为兼容 facade 的二级聚合模块依赖
- **THEN** 架构边界测试 MUST 失败
- **AND** 失败信息 MUST 指向推荐的窄模块路径

#### Scenario: inventory 更新
- **WHEN** 新增或拆分 scripts、tools、viewer support 或大型 domain helper
- **THEN** 项目表面积 inventory 或等价文档 MUST 记录该入口的 lifecycle 和职责
- **AND** 测试 allowlist MUST 与文档保持一致

### Requirement: 拆分后轻量导入保持
热点模块拆分 MUST 不破坏现有轻量导入边界。schema、constants、objective metadata 查询、dataset descriptor 查询和 path helper 查询 MUST 不因为拆分而导入训练循环、dataset 实例、模型、大型可视化依赖或真实数据读取逻辑。

#### Scenario: objective schema 轻量导入
- **WHEN** 开发者导入 objective metadata 的 schema/registry 子模块
- **THEN** 导入 MUST 成功
- **AND** 系统 MUST 不导入训练器、dataset、模型或 matplotlib

#### Scenario: dataset runtime schema 轻量导入
- **WHEN** 开发者查询 dataset descriptor 或 runtime schema helper
- **THEN** 查询 MUST 不打开 HDF5、CSV、image、LiDAR 或 checkpoint 文件
- **AND** 查询 MUST 不导入训练循环
