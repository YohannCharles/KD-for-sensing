## ADDED Requirements

### Requirement: Hist 组件注册已退役
组件注册表 MUST 不再注册 HiST-Beam/Hist 专用模型、loss、adapter、prototype 或 workflow 组件。旧注册名 MUST 被识别为已删除或未知名称，并给出当前支持范围。

#### Scenario: hist_beam_fusion 构建失败
- **WHEN** 用户请求构建 `hist_beam_fusion`
- **THEN** registry 或配置构建 MUST 拒绝该名称
- **AND** 错误信息 MUST 说明 Hist/HiST-Beam 研究线已退役

#### Scenario: Hist variants 不作为模型注册名
- **WHEN** 默认组件导入完成后开发者查看 `MODELS` 注册名
- **THEN** 注册名 MUST 不包含 HiST-Beam variants、P3/radio prototype variants、image-only Hist probe variants 或 history-anchor Hist variants
- **AND** 当前主线模型注册名 MUST 继续可用

### Requirement: 已删除组件错误包含 Hist 迁移方向
当用户引用 Hist 旧组件名时，错误信息 MUST 区分退役研究线与普通拼写错误，并指向当前推荐 workflow 或说明无兼容迁移。

#### Scenario: Hist 旧模型名错误可诊断
- **WHEN** 用户配置 `model.primary.type: hist_beam_fusion`
- **THEN** 系统 MUST 抛出包含 `hist_beam_fusion` 的错误
- **AND** 错误信息 MUST 提示使用当前 supervised、adapter、GPS candidate、residual fusion 或其它保留 workflow
