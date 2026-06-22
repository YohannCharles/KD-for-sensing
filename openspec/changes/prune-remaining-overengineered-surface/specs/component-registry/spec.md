## ADDED Requirements

### Requirement: Registry 不保活退役整模型类
组件 registry MUST 只保留当前 canonical 构建所需的 dataset、model、encoder、projector、core、head、loss、metric 和 preprocessor 名称。已经退役且不再注册的整模型类和旧 alias MUST 不通过直接导入测试或 facade 继续作为 current API。

#### Scenario: 退役模型 registry 构建失败
- **WHEN** 用户通过 `MODELS.build()` 请求已退役的 strong/lightweight/teacher/student 旧整模型名称
- **THEN** registry MUST 拒绝该名称
- **AND** 错误 MUST 使用现有 unknown-name 或保留的 removed-name 风格列出当前可用名称

#### Scenario: 当前组件仍可注册和构建
- **WHEN** 构建流程调用默认组件导入后构建 `modular_sequence`、当前 encoder、current fusion whole-model exception 或当前 loss/metric
- **THEN** registry MUST 保持变更前的构建行为
- **AND** 删除退役类 MUST 不影响这些当前注册名

### Requirement: Removed guard 只保留有迁移价值的名称
`register_removed()` 或等价 removed-name guard MUST 只用于仍可能从当前迁移路径触发、且普通 unknown-name 错误不足以防止误用的名称。完全退役、已有 OpenSpec tombstone 或只由测试 fixture 引用的名称 MUST 回落为 unknown-name，除非设计说明记录保留理由。

#### Scenario: 低价值 removed-name guard 被删除
- **WHEN** 某个 removed-name guard 只服务历史 fixture 或已退役研究线文案
- **THEN** 本 change MAY 删除该 guard
- **AND** 对应测试 MUST 改为验证 current registry 不注册该名称，而不是要求专属迁移文案

### Requirement: Registry helper 不新增自检抽象
Registry 的最小契约 MUST 由 focused tests 覆盖。项目 MUST 不保留或新增只包装测试逻辑的 registry self-check helper 作为 runtime API。

#### Scenario: 删除 registry self-check helper
- **WHEN** registry build、duplicate、unknown 和 missing parameter 行为已由 tests 覆盖
- **THEN** 本 change MUST 删除只服务这些检查的 runtime self-check helper
- **AND** 删除 MUST 不影响 registry 构建当前组件
