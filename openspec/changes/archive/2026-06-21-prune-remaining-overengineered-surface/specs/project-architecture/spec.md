## ADDED Requirements

### Requirement: 无价值 re-export facade 必须退出当前导入面
包级或子包级 facade 如果只 re-export owner 模块符号、没有独立行为、没有 current CLI/registry/config 依赖，且内部代码可直接导入 owner 模块，则项目 MUST 删除该 facade 或把它收缩为极薄 public shim。内部源码、测试和文档 MUST 使用真实 owner 路径，不得继续通过 facade 维持旧 import。

#### Scenario: 内部代码迁出 objective metadata facade
- **WHEN** 内部代码需要 prediction objective metadata helper
- **THEN** 代码 MUST 直接从 `kd_sensing.engine.objectives.metadata` 或对应 objectives owner 导入
- **AND** `kd_sensing.engine.objective_metadata` MUST 不再作为内部 helper import source

#### Scenario: data 和 datasets lazy export 不再扩展
- **WHEN** 代码需要 `DeepSense6GDataset`、`MMWDataset`、`SyntheticSequenceDataset`、sample helper 或 target-shot helper
- **THEN** 代码 MUST 从具体 owner 模块导入
- **AND** `kd_sensing.data` 或 `kd_sensing.data.datasets` MUST 不新增 lazy re-export 来保留旧路径

#### Scenario: fusion 旧类名 alias 不再作为迁移层
- **WHEN** 用户或测试导入已退役 fusion 旧类名 alias
- **THEN** 系统 MAY 使用普通 `AttributeError` 或 unknown import 失败
- **AND** 项目 MUST 不通过 `_REMOVED_ALIASES` 或等价 facade 表继续承诺每个退役类名的专属迁移错误

### Requirement: 内部模块不得使用星号导入表达公共 API
当前源码内部模块 MUST 使用显式 import 表达依赖。`from <module> import *` 只允许在明确保留的兼容 facade 中使用；业务 owner、runner、manifest、artifact、plot、scenario 或 predictive 模块 MUST 显式导入实际使用符号。

#### Scenario: JEPA benchmark 模块显式导入 common 符号
- **WHEN** 开发者修改 `src/kd_sensing/diagnostics/jepa_benchmark_*.py`
- **THEN** 修改后的 owner 模块 MUST 不新增 `from kd_sensing.diagnostics.jepa_benchmark_common import *`
- **AND** 需要的常量、类型和 helper MUST 显式列在 import 语句中

#### Scenario: public facade 保持薄层
- **WHEN** 某个模块被明确保留为 public facade
- **THEN** facade MAY re-export 稳定 API
- **AND** facade MUST 不承载 suite-specific helper 实现或成为内部模块的依赖来源

### Requirement: 小型类型别名文件不得替代标准类型
项目 MUST 不为单个 `dict[str, Any]`、`PathLike` 或等价标准类型保留独立 package 文件。只有跨多个 owner 共享且能减少真实复杂度的类型定义 MAY 保留；否则代码 MUST 直接使用标准类型或在真实 owner 模块中定义局部类型。

#### Scenario: 删除 AnyConfig 别名文件
- **WHEN** `_typing.AnyConfig` 只等价于 `dict[str, Any]` 且只有少量调用方
- **THEN** 本 change MUST 将调用方改为标准类型注解
- **AND** `src/kd_sensing/_typing.py` MAY 被删除
