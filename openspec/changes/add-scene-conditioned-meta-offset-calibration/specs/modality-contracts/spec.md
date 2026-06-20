## ADDED Requirements

### Requirement: Scenario conditioning 字段不是新模态
系统 MUST 将 `target_state`、`object_tokens`、`scene_params`、support metadata、domain key、scene/town/scenario/weather 字段定义为 conditioning、context、target 或 metadata 字段，而不是 canonical sensing modality。新增 scene meta-offset 功能 MUST 继续使用中心化模态契约中的 `image`、`radar`、`gps`、`lidar`、`mmwave` 和 `csi` 作为唯一 sensing modality 名称。

#### Scenario: target state 不进入 modality list
- **WHEN** 配置启用 target-conditioned prediction
- **THEN** `target_state` MUST 通过 batch/context mapping 传入支持该字段的模型
- **AND** `model.primary.modalities`、fusion slug 和 modality ordering MUST NOT 包含 `target_state`

#### Scenario: object tokens 不创建 object modality
- **WHEN** 配置启用 GT 或 detector object tokens
- **THEN** object tokens MUST 被记录为 context/conditioning input
- **AND** 系统 MUST NOT 注册 `object`、`bbox`、`detector_bbox` 或等价伪模态名称

#### Scenario: difficulty 与 scene shift 不创建伪模态
- **WHEN** synthetic 或真实配置声明 image weather corruption、GPS delay、modality reliability shift 或 missing modality sweep
- **THEN** affected sensing field MUST 继续解析为 canonical modality key
- **AND** 系统 MUST 拒绝 `image_hard`、`delayed_gps`、`radio_oracle` 或等价伪模态名称
