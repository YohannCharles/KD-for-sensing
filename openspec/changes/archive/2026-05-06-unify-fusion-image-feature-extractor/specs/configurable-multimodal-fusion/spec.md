## ADDED Requirements

### Requirement: Fusion teacher image 分支复用单模态特征提取器
`fusion_teacher` 在启用 image 模态时 MUST 使用 image-only teacher 暴露的 `ImageFeatureExtractor` 作为帧级特征提取器。系统 MUST 不再为 fusion teacher 维护单独的旧版 image feature extractor 副本。

#### Scenario: 构建包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 包含 `image`
- **THEN** 模型 MUST 将 `image_feature_extractor` 初始化为 `ImageFeatureExtractor`
- **AND** image 分支输出 MUST 保持 `[B, T, feature_size]` 形状以参与 fusion projection

#### Scenario: 构建不包含 image 的 fusion teacher
- **WHEN** 用户构建 `fusion_teacher` 且 `modalities` 不包含 `image`
- **THEN** 模型 MUST 不创建 image feature extractor
- **AND** 缺失 image 输入不得阻止该 fusion teacher forward

#### Scenario: 旧 fusion teacher checkpoint 结构不匹配
- **WHEN** 用户使用严格加载将旧 `FusionImageFeatureExtractor` 结构的 `fusion_teacher` checkpoint 加载到新模型
- **THEN** 系统 MUST 报告 checkpoint 结构不匹配
- **AND** 错误信息 MUST 包含 missing keys 或 unexpected keys 诊断
