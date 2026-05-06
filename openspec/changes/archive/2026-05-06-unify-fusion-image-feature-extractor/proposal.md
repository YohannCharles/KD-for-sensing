## Why

`fusion_teacher` 的 image 分支仍在 `fusion/networks.py` 中保留一份旧的 `FusionImageFeatureExtractor`，而其它 teacher 分支已经复用各自单模态模块。单模态 `ImageFeatureExtractor` 后续加入了 channel/spatial attention 和批量帧处理，fusion image 分支没有同步，导致同一 image 模态在 image-only 与 fusion teacher 中行为不一致，也增加后续维护成本。

## What Changes

- 让 `fusion_teacher` 的 image 分支复用 `kd_sensing.models.image.ImageFeatureExtractor`。
- 移除或停止使用 `FusionImageFeatureExtractor` 这份 fusion 内部旧副本，使 image、radar、GPS、LiDAR、mmWave teacher 分支都来自单模态 feature extractor。
- 为 fusion image 分支增加覆盖测试，确认启用 image 的 fusion teacher 能构建、forward，并保持输出契约 `(pred, features, output_features)`。
- 明确 checkpoint 影响：旧 `fusion_teacher` 权重若包含 `image_feature_extractor` 旧结构参数，严格加载时可能出现 missing/unexpected keys，必须依赖现有 checkpoint 诊断错误暴露，而不是静默兼容。
- **BREAKING**: 旧版 `fusion_teacher` checkpoint 的 image 分支参数结构可能不再与新模型匹配，需要重新训练或显式使用非严格加载进行迁移实验。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `configurable-multimodal-fusion`: 约束 `fusion_teacher` 启用 image 时必须复用 image-only teacher 的 `ImageFeatureExtractor`，避免维护独立旧副本。

## Impact

- 影响代码：`src/kd_sensing/models/fusion/networks.py`、相关模型导入/export、fusion 模型测试。
- 影响模型行为：启用 image 的 `fusion_teacher` 会获得与 image-only teacher 一致的 image frame feature extractor，包括 attention 模块和批量帧处理。
- 影响 checkpoint：旧 fusion teacher 权重可能需要重新训练；严格加载错误应继续通过现有 checkpoint 诊断机制展示 missing/unexpected keys。
- 不新增依赖，不改变数据集、预处理、训练循环或配置文件语义。
