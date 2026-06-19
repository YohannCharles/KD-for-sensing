## ADDED Requirements

### Requirement: 内部 helper 合并与边界保留
项目 MUST 允许同一 owner 下的内部 helper 在过度拆分、单调用点、低复用或只服务 re-export 时合并回清晰 owner 模块。合并 MUST 不新增旧入口、兼容聚合层、仓库根运行方式或跨领域 `utils` 聚合模块。合并后 public facade、console script、轻量导入边界、数据/配置/训练职责边界和本地产物边界 MUST 保持稳定。

#### Scenario: 合并同 owner 内部 helper
- **WHEN** 开发者将只被同一 owner 模块使用的内部 helper 文件合并回该 owner 模块
- **THEN** 包内公开 import、CLI 入口和 console script MUST 继续指向同一 public surface
- **AND** 合并后的实现 MUST 不要求调用方从旧 helper 文件导入符号
- **AND** 架构边界测试和治理索引 MUST 更新为合并后的 owner 文件布局

#### Scenario: 不创建新的兼容聚合层
- **WHEN** 开发者为了减少 Python 文件数调整模块布局
- **THEN** 系统 MUST 不新增只转发旧路径的兼容 facade、跨领域 `helpers.py`、仓库根脚本入口或绕过 `src/kd_sensing` 包结构的运行方式
- **AND** 已退役入口和 retired research line MUST 不因合并而恢复

#### Scenario: 轻量导入边界保持稳定
- **WHEN** 合并发生在 diagnostics、engine、preprocessing 或 baseline 内部模块
- **THEN** 导入轻量配置、路径工具、包级公共 API 或已登记 thin facade 时 MUST 不额外触发 dataset 读取、模型权重加载、训练逻辑或重型可视化依赖

### Requirement: 内部冗余检查可精简但外部边界检查保留
项目 MAY 删除内部私有 helper 中重复的 `assert`、重复类型检查、重复空值保护和只重新包装同类异常的 `try/except`，但用户输入、配置/manifest、文件路径、split/label-space/metric comparability、no-future-leak、输出产物边界和测试 fixture 契约相关检查 MUST 保留清晰失败模式。

#### Scenario: 删除内部重复检查
- **WHEN** 一个私有 helper 只由同 owner 调用，且调用方已经验证输入形状、类型或必需字段
- **THEN** 实现 MAY 删除该 helper 内重复的断言或二次类型检查
- **AND** focused tests MUST 证明正常路径输出、schema 和指标语义没有改变

#### Scenario: 保留用户输入边界检查
- **WHEN** CLI、manifest、配置文件、路径解析、数据 split、label space 或 checkpoint provenance 来自用户输入或外部文件
- **THEN** 系统 MUST 继续在边界处拒绝无效输入或记录明确 warning
- **AND** 错误或 warning MUST 足以定位无效字段、路径或不可比较原因

#### Scenario: 保留实验安全边界
- **WHEN** 代码处理 temporal source index、difficulty replay metadata、输出目录、cache、checkpoint 或真实实验产物
- **THEN** 系统 MUST 继续保证 no-future-leak、deterministic replay 和 ignored runtime artifact 边界
- **AND** 合并或删检查 MUST 不允许训练输出、日志、cache 或 checkpoint 进入源码变更
