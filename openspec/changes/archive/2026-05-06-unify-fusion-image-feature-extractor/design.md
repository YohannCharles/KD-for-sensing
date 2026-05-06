## Context

当前 `FusionModalityNet` 的 teacher 分支中，radar、GPS、LiDAR、mmWave 都直接复用单模态模块中的 feature extractor，只有 image 仍在 `fusion/networks.py` 内定义 `FusionImageFeatureExtractor`。这份旧实现与 `models/image.py` 中的 `ImageFeatureExtractor` 卷积主干相近，但缺少 channel/spatial attention，并且通过 Python 循环逐帧处理序列。

`ImageFeatureExtractor` 已经成为 image-only teacher 的公开实现和包导出名称。继续保留 fusion 专用 image encoder 会让同一 image 模态出现两套 teacher feature 定义，后续调参、测试和 checkpoint 排查都更容易产生误解。

## Goals / Non-Goals

**Goals:**

- 让 `fusion_teacher` 启用 image 时直接实例化 `ImageFeatureExtractor(feature_size, image_channels)`。
- 删除或停止使用 `FusionImageFeatureExtractor`，避免继续维护重复实现。
- 保持 `fusion_teacher` 的 forward 输入、输出和 `modalities` 配置语义不变。
- 增加测试覆盖，验证包含 image 的 fusion teacher 使用统一 image feature extractor 并能完成 forward。
- 保持 checkpoint 结构不匹配时的严格加载诊断，不做静默兼容。

**Non-Goals:**

- 不改变 `fusion_student` 的轻量 image 分支；student 仍使用 fusion student 自身的轻量 CNN 结构。
- 不调整 image-only teacher 的 GRU、temporal attention、classifier 或训练配置。
- 不新增 legacy 配置开关来继续选择旧 `FusionImageFeatureExtractor`。
- 不迁移旧 checkpoint 权重；旧权重如需使用，应重新训练或由用户显式选择非严格加载实验。

## Decisions

1. `fusion_teacher` 直接导入并使用 `ImageFeatureExtractor`。

   这样 image、radar、GPS、LiDAR、mmWave teacher 分支都遵循“fusion teacher 复用单模态 feature extractor”的结构。替代方案是把 attention 逻辑复制进 `FusionImageFeatureExtractor`，但这仍会保留重复实现，未来再次漂移。

2. 不保留旧 image encoder 的运行时选择开关。

   旧结构只服务历史 checkpoint 兼容，继续暴露开关会扩大配置矩阵和测试成本。当前项目已经默认严格加载 checkpoint；结构不匹配时，缺失和多余 key 会直接暴露。替代方案是新增 `legacy_image_fusion_encoder: true`，但它会让用户继续在两套 image teacher 语义之间切换，不利于后续实验对齐。

3. 测试关注结构选择和输出契约。

   最小覆盖应断言 `FusionModalityNet(..., modalities=["image", ...])` 的 `image_feature_extractor` 是 `ImageFeatureExtractor`，并使用小 batch 检查 forward 返回三元组形状。由于 `ImageFeatureExtractor` 的 FC 层固定依赖 `224x224` 输入，测试输入应使用 `[B, T, C, 224, 224]` 或沿用项目现有构造 helper。

4. 文档只在必要位置说明 checkpoint 影响。

   这次变更不改变配置入口和训练命令，不需要大范围文档重写。若 README 或兼容性说明已有 checkpoint 结构不匹配诊断描述，可不重复；若测试或变更说明需要提到 breaking impact，应明确旧 fusion teacher 权重可能需要重训。

## Risks / Trade-offs

- [Risk] 旧 `fusion_teacher` checkpoint 无法严格加载 -> Mitigation：依赖现有严格加载诊断展示 missing/unexpected keys，并在变更说明中明确需要重训或显式非严格加载。
- [Risk] fusion teacher 指标与旧实验不可直接比较 -> Mitigation：把该变更标记为 breaking，并要求新的 fusion teacher baseline 在统一 encoder 后重新训练。
- [Risk] 训练耗时或显存轻微变化 -> Mitigation：`ImageFeatureExtractor` 使用批量帧展开，通常比逐帧 Python 循环更稳定；用单元测试覆盖形状，必要时后续用 profile 评估吞吐。
- [Risk] 用户误以为 fusion student 也会复用 image-only student -> Mitigation：规格明确本变更只约束 `fusion_teacher` 的 image feature extractor，student 轻量分支不在范围内。
