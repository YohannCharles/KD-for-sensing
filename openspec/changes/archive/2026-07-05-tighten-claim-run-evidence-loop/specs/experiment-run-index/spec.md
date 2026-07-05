## ADDED Requirements

### Requirement: Run index 支持 run card provenance
Run index MUST 暴露 run card 所需的只读 provenance 字段，包括 run state、status artifact、metrics path、config path、checkpoint summary、output root、started/completed timestamps 和 warnings。Run index MUST 不移动、删除或重写本地运行产物。

#### Scenario: run card 查询 run index
- **WHEN** run card builder 查询 run index 中的某个 run
- **THEN** run index MUST 返回可用于 provenance 的路径和状态摘要
- **AND** 如果 metrics、config 或 checkpoint 缺失，结果 MUST 以 warning 表达而不是伪造字段
