## ADDED Requirements

### Requirement: 低价值聚合面必须收敛
项目 MUST 避免新增或保留没有 current public 契约价值的 package-level barrel、兼容 facade、单用途包装模块或重复 helper 聚合。仓库内部实现 MUST 直接导入职责明确的 owner module；只有 README、current spec、pyproject console script 或明确 public import 契约登记的路径才可保留薄入口。收敛实现 MUST 不恢复旧脚本入口、退役研究线入口或绕过 `src/kd_sensing` 包结构的运行方式。

#### Scenario: 内部实现不依赖 package barrel
- **WHEN** 包内训练、评估、诊断、数据或测试代码需要使用具体 owner 功能
- **THEN** 代码 MUST 从职责明确的 owner module 导入
- **AND** 代码 MUST 不为了少写 import 路径而依赖 package `__init__.py` 的重依赖 re-export

#### Scenario: 单用途包装模块被合并或登记
- **WHEN** 某个源码模块只包装一个 owner 函数、类或常量，且没有 current public import 契约
- **THEN** 实现 MUST 将该包装合并回调用点或 owner module
- **AND** 若暂缓合并，inventory MUST 登记其保留原因、owner、删除条件和 focused validation

#### Scenario: 重复 helper 不形成新杂物间
- **WHEN** 多个模块出现语义一致的 CSV、JSON、float、slug 或 path helper
- **THEN** 实现 MUST 优先复用已有 owner helper 或建立领域窄 helper
- **AND** 实现 MUST 不把这些 helper 放入会扩大轻量导入面的跨领域大 `utils` 聚合

#### Scenario: 收敛不触碰本地产物
- **WHEN** 开发者实施聚合面收敛、包装删除或 helper 合并
- **THEN** 变更 MUST 不删除、移动、压缩或重写 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`All_models/` 或其它本地运行产物
- **AND** 变更 MUST 只影响源码、测试、文档、配置或 OpenSpec artifact
