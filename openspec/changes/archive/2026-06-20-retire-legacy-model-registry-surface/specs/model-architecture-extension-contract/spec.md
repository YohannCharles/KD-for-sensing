## ADDED Requirements

### Requirement: Legacy whole-model baseline retirement
普通 supervised/adaptation baseline 的 legacy whole-model 注册名 MUST 可退役为 removed guard。退役后，canonical config 路径 MAY 保留原文件名和 run name，但 `model.primary.type` MUST 使用 `modular_sequence` 或明确保留的 whole-model exception。

#### Scenario: single-modality strong/lightweight 退役
- **WHEN** canonical image、radar、GPS、LiDAR 或 mmWave strong/lightweight/supervised 配置仍作为 current 入口保留
- **THEN** 配置 MUST 使用 `model.primary.type: modular_sequence`
- **AND** 对应旧 whole-model 注册名 MUST NOT 出现在 current `MODELS.list()` 中

#### Scenario: legacy whole-model 名称必须有迁移 guard
- **WHEN** 用户请求本 change 退役的旧 whole-model 注册名
- **THEN** registry MUST 抛出 removed guard 错误
- **AND** 错误信息 MUST 指出等价或推荐的 `modular_sequence` 组件组合

#### Scenario: whole-model exception 保持显式
- **WHEN** 模型仍以完整 `MODELS.register(...)` 形式保留
- **THEN** 该模型 MUST 属于 workflow/paper reproduction、current spec 明确能力或 active design 记录的 whole-model exception
- **AND** 模型 MUST 继续提供可审计 metadata、registry build test 和 forward/output adaptation 覆盖

### Requirement: Model registry inventory reflects lifecycle
人类可读模型架构目录和机器可读维护索引 MUST 区分 current、removed guard 和 deferred cleanup。退役名称不得与 current 名称混列。

#### Scenario: 架构目录只展示 current 模型
- **WHEN** 维护者查看 `docs/model_architecture_inventory.md`
- **THEN** current model/encoder/core/head 表格 MUST 不包含 removed guard 名称
- **AND** 退役名称 MUST 只出现在退役边界或 migration 表中

#### Scenario: allowlist 与 registry 同步
- **WHEN** 架构边界测试读取 `docs/maintainer_context_index.yaml` 的 model registration allowlist
- **THEN** allowlist MUST 只包含 current `MODELS` 注册名
- **AND** removed guard 名称 MUST 被单独记录或通过测试断言其不可构建
