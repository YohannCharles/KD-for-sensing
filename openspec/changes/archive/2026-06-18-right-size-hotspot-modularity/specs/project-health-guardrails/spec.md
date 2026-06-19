## MODIFIED Requirements

### Requirement: 维护性热点 inventory
项目 MUST 维护一份可审计的维护性热点 inventory，记录已知超长模块、超长函数、超长类、兼容 facade、推荐拆分方向、合并/收敛方向、暂缓原因和右尺寸化预算策略。新增热点或热点显著扩大时，项目 MUST 更新 inventory、拆分到稳定窄模块、合并低价值边界，或登记有验证命令支撑的理由化例外。

#### Scenario: 已知热点被命名
- **WHEN** 开发者审阅项目健康 inventory
- **THEN** inventory MUST 记录当前已知热点的文件路径、符号名、热点类型、当前规模指标、推荐拆分方向、合并/收敛方向或接受当前尺寸的理由
- **AND** inventory MUST 包含训练主循环、DeepSense6G/MMW dataset、BeamBench Image AE+GPS workflow、run index、evaluation pass、batch preparation 和 manifest builder 等当前高维护成本区域

#### Scenario: 静态检查发现未登记热点
- **WHEN** 架构边界或健康检查发现新增超长函数、超长类或 facade 回流
- **THEN** 检查 MUST 失败或输出明确失败信息
- **AND** 失败信息 MUST 指向更新 inventory、拆分到窄模块、合并低价值边界或增加有理由例外这几种修复路径之一

#### Scenario: facade 预算继续硬失败
- **WHEN** 已登记为 facade 或公开兼容入口的模块超过硬预算，或重新承载已迁出的 suite-specific helper 实现
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求将实现移回职责明确的窄模块或删除不再需要的兼容 facade

#### Scenario: 业务热点允许理由化 headroom
- **WHEN** 已登记业务 workflow、dataset 或 diagnostic analysis 模块略超预算但处于索引声明的 headroom 内
- **THEN** 健康检查 MAY 接受该状态
- **AND** 索引和 inventory MUST 提供 rationale、validation commands 和后续动作分类，例如 `monitor`、`split-next`、`right-size-accepted` 或 `merge-candidate`

#### Scenario: 低价值边界被合并或登记为合并候选
- **WHEN** 维护者发现单调用点包装类、只为减少行数的 helper 模块、重复 `utils` 聚合或无公开兼容价值的 facade
- **THEN** 实现 MUST 将其合并回清晰 owner、改为私有局部 helper，或在热点 inventory 中登记为 `merge-candidate`
- **AND** 合并 MUST 不把实现重新堆回公开 facade 或绕过当前包结构

#### Scenario: 右尺寸化检查不改变 runtime
- **WHEN** 开发者运行架构边界或健康检查
- **THEN** 检查 MUST 只读取已跟踪源码、配置、文档、OpenSpec artifact、pyproject 和测试文件
- **AND** 检查 MUST 不读取真实 `dataset/`、`outputs/`、`logs/`、checkpoint、cache 或 TensorBoard event

## ADDED Requirements

### Requirement: 高风险源码表面修复按 wave 管理
项目 MAY 对热点模块执行高风险结构重构，但该重构 MUST 按 remediation wave 管理。每个 wave MUST 记录目标文件、owner 边界、公开 import/CLI 保持策略、focused validation commands 和回滚条件。高风险 wave MUST 不把训练数学语义、数据 split 语义、beam label 口径、checkpoint schema 或默认输出目录作为隐式变更。

#### Scenario: wave 开始前捕获 baseline
- **WHEN** 维护者开始一个高风险热点修复 wave
- **THEN** tasks 或实现说明 MUST 记录该 wave 的目标文件、当前热点规模、公开入口和最小 focused tests
- **AND** 若已有测试红点，说明 MUST 区分既有红点和本 wave 引入的新红点

#### Scenario: wave 完成后独立验证
- **WHEN** 一个 wave 完成源码移动、拆分或合并
- **THEN** 维护者 MUST 运行该 wave 对应的 focused tests 和 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- **AND** 未运行的验证 MUST 在最终说明中记录原因和剩余风险

#### Scenario: 小而内聚模块不被强制拆分
- **WHEN** 模块低于热点阈值、职责内聚且无重复抽象或公开边界问题
- **THEN** 健康护栏 MUST NOT 要求仅因相邻热点修复而拆分该模块
- **AND** 维护者 MAY 只补测试或登记为 monitor
