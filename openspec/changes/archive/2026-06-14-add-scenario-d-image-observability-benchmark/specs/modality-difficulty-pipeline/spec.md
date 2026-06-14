## ADDED Requirements

### Requirement: Image observability transform
系统 MUST 提供 image observability transform，用于在统一 difficulty pipeline 中确定性调制 image modality。Transform MUST 支持 frame dropout、burst dropout、weather、blur、occlusion、low-light、seed、valid mask、observability score 和 replay metadata。

#### Scenario: ImageObservabilityTransform 参数
- **WHEN** 开发者构建 image observability transform
- **THEN** transform MUST 支持 `image_dropout_prob`、`image_burst_dropout_prob`、`max_burst_len`、`image_weather_severity`、`image_blur_prob`、`image_occlusion_prob`、`image_occlusion_ratio`、`image_lowlight_prob` 和 `seed`
- **AND** 参数 MUST 可由 difficulty profile/operator config 或 Scenario D preset 标准化生成

#### Scenario: corruption 与 missing 区分
- **WHEN** transform 只应用 weather、low-light、blur 或 occlusion
- **THEN** `image_valid_mask` MUST 保持有效，除非配置另行声明整帧不可用
- **AND** transform MUST 写出 corruption type、severity、frame range 和 operator parameters
- **AND** GPS、target label、sample id 和 split metadata MUST 保持不变

#### Scenario: dropout 生成 invalid mask
- **WHEN** transform 应用 frame dropout 或 burst missing
- **THEN** 系统 MUST zero-fill、mask-fill 或使用配置声明的 missing token 表达缺失 image
- **AND** 系统 MUST 写出 `image_valid_mask`、`image_dropout_mask` 或 `image_burst_dropout_mask`
- **AND** 缺失表达方式 MUST 写入 warnings 或 replay metadata

### Requirement: Image observability score
系统 MUST 计算 `image_observability_score`，用于表达当前 image 输入的可用性。Score MUST 由 dropout、blur、occlusion、low-light 和 weather severity 等输入退化因素确定，MUST 位于可解释范围内，并 MUST 不作为 target supervision。

#### Scenario: score 随退化降低
- **WHEN** image observability transform 应用更高 dropout、blur、occlusion 或 low-light severity
- **THEN** `image_observability_score` MUST 不高于 clean condition 的 score
- **AND** score metadata MUST 记录参与计算的 corruption factors

#### Scenario: score 可被 batch 和模型消费
- **WHEN** difficulty pipeline 输出 batch
- **THEN** batch MUST 包含 `image_observability_score` 或等价 metadata 字段
- **AND** 训练、评估或 benchmark runtime MUST 能将该字段传递给支持 observability-aware fusion 的模型

### Requirement: Scenario D preset 复用 difficulty pipeline
Scenario D 的 `D0` 到 `D7` preset MUST 通过 shared difficulty profile/operator registry 解析和执行。Benchmark、evaluation 和 train-time augmentation 使用相同 profile id、condition、severity、seed、split 和 sample id 时，MUST 产生一致的 image corruption、mask 和 replay metadata。

#### Scenario: 同 seed 可复现
- **WHEN** synthetic image batch 使用同一 Scenario D condition、seed 和 sample id 两次应用 transform
- **THEN** 两次输出的 image tensor、valid/dropout mask、observability score 和 metadata MUST 一致
- **AND** target label 和 sample id MUST 与输入一致

#### Scenario: unknown D-level 被拒绝
- **WHEN** profile 或 manifest 引用未知 image observability level `D9_magic`
- **THEN** 系统 MUST 拒绝配置
- **AND** 错误信息 MUST 包含未知 condition 和可用 D-level 列表
