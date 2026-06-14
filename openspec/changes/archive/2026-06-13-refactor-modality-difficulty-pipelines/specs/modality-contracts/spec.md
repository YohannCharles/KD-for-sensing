## ADDED Requirements

### Requirement: Difficulty profile 复用 canonical modality keys
Difficulty profile MUST 使用中心化模态契约中的 canonical modality name、sample key 和 fusion input key 来声明 affected modality。难度 profile MUST 不新增 `gps_noisy`、`delayed_gps`、`image_hard` 等伪模态名称，也 MUST 不要求训练、评估或模型 forward 为每种难度新增专用输入分支。

#### Scenario: GPS difficulty 使用 gps canonical key
- **WHEN** profile 声明 GPS jitter、delay 或 dropout
- **THEN** affected modality MUST 标准化为 `gps`
- **AND** transform MUST 作用于当前 batch 的 GPS sample key，并由现有 `gps_batch` 准备路径消费

#### Scenario: 拒绝伪模态名称
- **WHEN** 用户在 modalities 或 difficulty affected modality 中配置 `delayed_gps`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 指向 canonical modality `gps` 和 difficulty profile 配置

### Requirement: Difficulty mask 与 metadata 字段语义
模态契约或等价中心化 helper MUST 定义 difficulty 产生的输入相关 mask/metadata 字段语义。GPS async/missing 字段至少 MUST 覆盖 valid、stale、delay steps、source index 和 dropout mask；image degradation 字段至少 MUST 覆盖 degradation type、severity、seed、frame range 和 optional mask。字段命名 MUST 避免与 target schema、auxiliary target 或 sensitive supervision 字段混淆。

#### Scenario: GPS async metadata 可查询
- **WHEN** 开发者查询 GPS modality 的 difficulty metadata fields
- **THEN** 系统 MUST 返回 `gps_valid_mask`、`gps_stale_mask`、`gps_delay_steps`、`gps_source_index` 和 `gps_dropout_mask` 或等价字段说明
- **AND** 这些字段 MUST 被标记为输入 reliability metadata，而不是 target supervision

#### Scenario: image degradation metadata 不改变 profile
- **WHEN** image difficulty operator 输出 degradation metadata
- **THEN** metadata MUST 记录为 difficulty metadata
- **AND** image input profile MUST 仍保持 `rgb_imagenet` 或配置解析后的当前 profile
