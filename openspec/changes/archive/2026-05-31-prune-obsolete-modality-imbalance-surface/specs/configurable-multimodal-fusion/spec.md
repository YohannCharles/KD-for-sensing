## ADDED Requirements

### Requirement: Canonical fusion virtual config 不扩展 legacy KD 模式
Canonical fusion virtual config 生成器 MUST 聚焦当前 no-KD strong/lightweight/fusion 主线和仍被 active specs 批准的 objective/snapshot 入口。生成器 MUST 不再把 `logits_kd` 或 `rkd` 作为所有 fusion modality slug 的 canonical virtual mode；不存在实体 YAML 的 legacy KD fusion 路径 MUST 清晰失败。

#### Scenario: no-KD fusion virtual config 继续可用
- **WHEN** 用户请求当前支持的 no-KD canonical fusion virtual config
- **THEN** 配置加载器 MUST 生成完整配置
- **AND** 生成配置 MUST 包含当前主线模型、数据、训练和 lineage metadata
- **AND** 生成配置 MUST 不包含 KD-only temperature、alpha 或 RKD 权重字段

#### Scenario: KD fusion virtual config 不再接管路径
- **WHEN** 用户请求不存在实体 YAML 的 fusion `logits_kd` 或 `rkd` 配置
- **THEN** 配置加载器 MUST 拒绝该路径
- **AND** 错误信息 MUST 指向 legacy KD virtual alias 已退役，而不是生成或替换为其它配置

### Requirement: Legacy KD baseline 不影响 canonical 模态 slug 解析
删除 fusion KD virtual modes 后，canonical 模态 slug 解析 MUST 继续支持当前合法模态集合、顺序规范化、重复模态拒绝、未知模态拒绝和单模态转发建议。

#### Scenario: canonical slug 校验保持稳定
- **WHEN** 用户请求 no-KD fusion virtual config，并使用合法模态集合
- **THEN** 系统 MUST 按固定模态顺序解析 slug 并生成配置
- **AND** 重复模态、未知模态或可转为单模态配置的路径 MUST 继续给出清晰错误或建议
