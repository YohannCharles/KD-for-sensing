## ADDED Requirements

### Requirement: Difficulty pipeline 内部依赖必须走 owner module
模态 difficulty pipeline 的内部调用方 MUST 直接依赖 schema、pipeline、operator registry 或 concrete operator owner module。`kd_sensing.data.difficulty` 子包根只可作为轻量 package marker 或明确外部 shim，不得承载内部 re-export barrel。

#### Scenario: schema 和 pipeline 显式导入
- **WHEN** engine、diagnostics、config、tests 或其它内部模块需要 difficulty profile、operator plan、pipeline builder 或 schema 类型
- **THEN** 它们 MUST 从 `kd_sensing.data.difficulty.schema` 或 `kd_sensing.data.difficulty.pipeline` 导入
- **AND** 它们 MUST 不从 `kd_sensing.data.difficulty` package root 获取这些符号

#### Scenario: operator 注册包例外保留
- **WHEN** default component import 流程需要注册 built-in difficulty operators
- **THEN** 它 MAY 导入 `kd_sensing.data.difficulty.operators` 以触发注册 side effect
- **AND** 该例外 MUST 不允许 `kd_sensing.data.difficulty` package root eager import operators 或重依赖

#### Scenario: difficulty package marker 保持轻量
- **WHEN** 后续 change 修改 `src/kd_sensing/data/difficulty/__init__.py`
- **THEN** 该文件 MUST 保持轻量 marker 或明确登记的 public shim
- **AND** 它 MUST 不为了减少内部 import 行数而重新 re-export schema、pipeline 或 operator 符号
