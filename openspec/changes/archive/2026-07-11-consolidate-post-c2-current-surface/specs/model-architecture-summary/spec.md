## ADDED Requirements

### Requirement: Architecture summary 只保留 instance/startup supporting surface
模型架构摘要 MUST 作为 training startup、U-Mask/AMR/AMBER focused validation 和 Scene31-34 profile 的 supporting owner 保留。它 MUST 不再提供 standalone CLI、candidate sweep ingestion、config override preflight、Markdown/CSV renderer 或独立 report 产品面。

#### Scenario: Current consumer 继续读取 instance summary
- **WHEN** training startup 或 current focused test 对已构建模型生成摘要
- **THEN** helper MUST 返回稳定、JSON 可序列化的 instance parameter/component schema
- **AND** Scene31-34 profile MUST 继续能读取 startup artifact 中的 architecture summary

#### Scenario: Retired summary surface 不存在
- **WHEN** 用户请求旧 architecture-summary CLI、sweep manifest renderer 或 config-only preflight
- **THEN** 对应入口和实现 MUST 不属于 current surface
- **AND** 项目 MUST 不新增 replacement renderer 或 wrapper

## MODIFIED Requirements

### Requirement: 统一模型架构摘要 schema
系统 MUST 为已构建模型实例和 training startup artifact 提供统一架构摘要 schema。摘要 MUST 是 JSON 可序列化对象，并 MUST 包含 `schema_version`、`source`、`model`、`parameters`、`components`、`warnings` 和 `comparability` 顶层字段；系统 MUST 不要求同一 owner 解析 sweep candidate 或任意 config-only 输入。

#### Scenario: 摘要包含稳定顶层字段
- **WHEN** current consumer 对一个已构建模型实例生成架构摘要
- **THEN** 摘要 MUST 包含 `schema_version`、`source.kind`、`model.registry_type`、`model.class`、`parameters.total_params`、`parameters.trainable_params`、`parameters.frozen_params`、`components` 和 `warnings`
- **AND** 摘要 MUST 能被 `json.dumps()` 序列化

#### Scenario: 摘要来源是实际实例
- **WHEN** 摘要来自真实 `nn.Module` 实例
- **THEN** `source.kind` MUST 为 `instance`
- **AND** `parameters.parameter_count_source` MUST 为 `actual_module` 或等价实际实例来源

### Requirement: 组件角色与参数分组
系统 MUST 按已构建模型的组件角色汇总参数量。对于 `modular_sequence`，摘要 MUST 至少识别 `encoders.<modality>`、`projectors.<modality>`、`representation_core` 和 `heads.<name>`；对于 current JEPA/TinyViT/ResNet 视觉组件，摘要 MUST 支持 image encoder 与 visual/context encoder 的实际实例参数分组。

#### Scenario: modular_sequence 组件分组
- **WHEN** current consumer 对 image+GPS `modular_sequence` 模型生成摘要
- **THEN** `components` MUST 包含 image encoder、GPS encoder、representation core 和 beam head 对应条目
- **AND** 每个条目 MUST 包含 path、class、semantic_role、total params 和 trainable params

#### Scenario: current visual context encoder 分组
- **WHEN** 已构建 current 模型包含 JEPA mean context、TinyViT 或 ResNet image encoder
- **THEN** 摘要 MUST 报告 image encoder params
- **AND** 能识别时 MUST 单独报告 visual/context encoder params

#### Scenario: 未知组件保留总数
- **WHEN** 系统无法识别某个模块的语义角色
- **THEN** 摘要 MUST 将其标记为 `unknown_component` 或等价 role
- **AND** 该模块参数 MUST 仍计入模型总参数

## REMOVED Requirements

### Requirement: 配置和 override 预检 warning
**Reason**: Config-only preflight 与 encoder/config validation owner 重复，且只服务已退役 standalone summary CLI。
**Migration**: Config loader/validation 继续负责非法 option 与 checkpoint policy；instance summary 只报告已构建模型。

#### Scenario: Config preflight 不再由 summary owner 提供
- **WHEN** 用户提交非法 encoder override 或 checkpoint 配置
- **THEN** canonical config validation MUST 处理该错误或 warning
- **AND** architecture summary MUST 不维护第二套 config parser

### Requirement: 架构摘要 CLI 和输出格式
**Reason**: Standalone CLI、sweep renderer 与多格式 report 没有 current consumer；startup JSON 和 Scene31-34 profile 已覆盖实际用途。
**Migration**: 使用 training `startup_summary.json` 和 protected Scene31-34 profile；历史 renderer 从 git/archive 查询。

#### Scenario: Standalone architecture summary 退出
- **WHEN** console scripts 和 module CLI 被枚举
- **THEN** architecture-summary command、candidate sweep renderer 和独立 output report MUST 不存在
- **AND** instance/startup summary helper MUST 继续可用
