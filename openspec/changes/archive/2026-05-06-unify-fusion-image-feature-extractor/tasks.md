## 1. 模型实现

- [x] 1.1 在 `src/kd_sensing/models/fusion/networks.py` 中导入 `ImageFeatureExtractor`。
- [x] 1.2 将 `FusionModalityNet` 的 image 分支改为实例化 `ImageFeatureExtractor(feature_size, image_channels)`。
- [x] 1.3 删除未使用的 `FusionImageFeatureExtractor` 类和相关冗余代码，确认 fusion teacher 其它模态分支保持不变。
- [x] 1.4 检查 `src/kd_sensing/models/__init__.py` 和 backbones 导出，无需新增重复导出或循环导入。

## 2. 测试覆盖

- [x] 2.1 新增或更新 fusion 模型测试，断言包含 image 的 `fusion_teacher` 使用 `ImageFeatureExtractor`。
- [x] 2.2 新增或更新 forward 形状测试，使用 `[B, T, C, 224, 224]` image 输入验证包含 image 的 fusion teacher 返回 `(pred, features, output_features)`。
- [x] 2.3 覆盖不包含 image 的 fusion teacher 构建路径，确认不会创建 image feature extractor 且不要求 image 输入。

## 3. 验证

- [x] 3.1 使用 `conda run -n kd_mm_beam pytest` 运行相关模型测试。
- [x] 3.2 使用 `conda run -n kd_mm_beam openspec status --change unify-fusion-image-feature-extractor` 或等价 OpenSpec 命令确认任务状态和规格文件有效。
- [x] 3.3 若测试暴露旧 checkpoint 加载差异，确认错误包含 missing keys 或 unexpected keys，而不是静默加载。
