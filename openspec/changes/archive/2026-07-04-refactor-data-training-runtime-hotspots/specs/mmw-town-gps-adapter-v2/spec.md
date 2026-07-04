## ADDED Requirements

### Requirement: MMW GPS v2 workflow 必须拆分 protocol 与 artifact writer
MMW Town GPS v2 workflow MUST 将运行上下文准备、label-space 解析、样本加载、support selection、protocol 执行、summary row 构造和 artifact 写出拆分为独立职责。

#### Scenario: artifact schema 兼容
- **WHEN** MMW GPS v2 workflow is refactored
- **THEN** prediction rows, summary rows, support rows, theta rows, branch rows, logits exports and metadata keys MUST remain compatible
- **AND** 默认输出目录语义 MUST 不改变
