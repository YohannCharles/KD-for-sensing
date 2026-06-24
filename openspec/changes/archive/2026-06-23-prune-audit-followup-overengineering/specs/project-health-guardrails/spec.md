## ADDED Requirements

### Requirement: 复杂度瘦身回流检查
项目健康护栏 MUST 能发现源码表面积中重新引入的低价值 package barrel、兼容 facade、重复 helper 聚合和 tracked runtime artifact。检查 MUST 只读取 tracked source、pyproject、README、docs、OpenSpec 和测试文件，不得扫描真实 `dataset/`、ignored `outputs/`、`logs/`、cache、checkpoint 或未跟踪本地 bytecode。

#### Scenario: 重依赖 barrel 回流被拒绝
- **WHEN** 已跟踪源码新增或扩大 package `__init__.py`，并 eager re-export 会导入 dataset、model、diagnostics、checkpoint registry、torch、pandas、matplotlib 或其它重依赖模块的符号
- **THEN** 架构边界检查 MUST 失败或要求该 re-export 有 current public 契约和轻量导入验证
- **AND** 失败信息 MUST 指向 owner module 直接导入或延迟导入收缩

#### Scenario: 兼容 facade 回流被拒绝
- **WHEN** 已删除的 builder、transform、BeamBench 聚合 owner、旧脚本 thin alias 或退役研究线 facade 重新出现在 tracked source 中
- **THEN** 架构边界检查 MUST 失败
- **AND** 失败信息 MUST 要求删除 facade 或在 active OpenSpec change 中登记 current public 契约

#### Scenario: 重复 helper 聚合需要理由
- **WHEN** 新增跨领域 helper 模块只收纳 CSV、JSON、float、slug 或 path 小工具，且没有明确 owner 和两个以上 current 调用点
- **THEN** 健康检查 MUST 失败或要求在 inventory 中登记为 `merge-candidate`、说明 owner 和验证命令
- **AND** 检查 MUST 不要求把领域私有 helper 强行移入全局 `utils`

#### Scenario: tracked runtime artifact 继续拒绝
- **WHEN** `__pycache__`、`.pyc`、`.pytest_cache`、`outputs/`、`logs/`、cache、checkpoint、TensorBoard event 或非允许权重文件被纳入 git tracked 文件
- **THEN** 源码表面积检查 MUST 失败
- **AND** 未跟踪或 ignored 的同类本地产物 MUST 不驱动常规架构边界测试失败
