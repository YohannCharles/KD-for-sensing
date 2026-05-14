## 1. Image ResNet-18 默认配置

- [x] 1.1 更新默认 image teacher/no-KD 配置，使其使用 `modular_sequence` + `resnet18_imagenet_rgb`，并配置 `rgb_imagenet`、`pretrained: true`、`weights: DEFAULT`、保守 freeze/unfreeze 策略。
- [x] 1.2 移除旧从头训练小 CNN image 配置入口，并确保 image student/KD 配置不再构建旧 image encoder。
- [x] 1.3 更新包含 image 的 canonical fusion teacher 配置或生成逻辑，使 image branch 默认使用与 image-only baseline 一致的 ResNet-18 encoder profile。
- [x] 1.4 确保 ResNet-18 encoder 的训练策略 metadata 进入 `final_config.yaml`、运行 metadata 或模型 metadata。

## 2. LiDAR Baseline 预处理修复

- [x] 2.1 更新默认 LiDAR teacher/no-KD baseline 配置，显式启用 `lidar_normalization.enabled: true` 和 `mode: streaming_stats`。
- [x] 2.2 确保 LiDAR train split 只在训练集上 fit normalizer，并将 normalizer/stats 工件保存到运行目录。
- [x] 2.3 确保 LiDAR test/eval split 复用训练阶段保存或传递的 normalizer，不在 test split 上重新 fit。
- [x] 2.4 更新默认 LiDAR baseline 配置的 cache 参数，确保 BEV size、ROI、FoV、ground/background 参数变化时 cache 隔离可追踪。
- [x] 2.5 检查默认 LiDAR ROI/FoV 配置，确认 Scenario 31/32 的 BEV 不是大量全零；必要时新增显式 FoV/ROI profile 作为 baseline 配置。

## 3. LiDAR 质量诊断与退化基线

- [x] 3.1 实现 LiDAR BEV 输入质量摘要，至少包含非空帧比例、每通道均值、标准差和零值比例。
- [x] 3.2 将 LiDAR 输入质量摘要写入训练/评估报告，并记录对应 split、ROI、FoV、cache 和 normalizer 参数。
- [x] 3.3 实现 majority-class baseline 统计，输出每个 future horizon 的 majority Top-1。
- [x] 3.4 实现 last-beam baseline 统计，在历史 beam label 可用时输出每个 future horizon 的 last-beam Top-1/Top-3。
- [x] 3.5 在 LiDAR 模型未超过 majority-class baseline 或输入质量异常时，在报告中标记 degradation risk。

## 4. 训练与评估报告

- [x] 4.1 扩展训练/评估 metadata，记录 image encoder profile、预训练权重、freeze 策略、LiDAR preprocessing profile 和 baseline 对比。
- [x] 4.2 确保 `teacher_metrics.json` 或等价报告包含 per-horizon Top-1/Top-3、跨 horizon 平均指标和退化基线字段。
- [x] 4.3 在文档或运行说明中明确 TensorBoard `accuracy/val` 与 per-horizon/平均指标的区别，避免把第一个 horizon 曲线误当论文表格指标。

## 5. 测试覆盖

- [x] 5.1 新增或更新配置测试，使用 `conda run -n kd_mm_beam pytest tests/test_resnet18_image_architecture.py tests/test_student_configs.py` 验证默认 image teacher/no-KD 使用 ResNet-18 预训练 encoder。
- [x] 5.2 新增或更新 fusion 配置测试，使用 `conda run -n kd_mm_beam pytest tests/test_fusion_image_feature_extractor.py tests/test_subset_specs.py` 验证包含 image 的 canonical fusion teacher 选择 ResNet-18 profile。
- [x] 5.3 新增或更新 LiDAR 配置测试，使用 `conda run -n kd_mm_beam pytest tests/test_lidar_modality.py tests/test_preprocessing_formats.py` 验证默认 LiDAR teacher/no-KD 显式启用 streaming stats normalizer 和可追踪 cache/ROI 参数。
- [x] 5.4 新增 LiDAR 诊断测试，使用 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 或新增测试文件验证输入质量摘要、majority baseline 和 degradation risk 字段。
- [x] 5.5 运行项目相关回归测试：`conda run -n kd_mm_beam pytest`。

## 6. 实验验证

- [x] 6.1 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/image/teacher_no_kd.yaml` 跑默认 image teacher baseline 的短训练或完整训练，确认构建的是 ResNet-18 ImageNet encoder。
- [x] 6.2 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/lidar/teacher_no_kd.yaml` 跑 LiDAR baseline 短训练，确认 normalizer fit、cache 使用和质量诊断均写入运行目录。
- [x] 6.3 对 Scenario 31 image/LiDAR 单模态输出做 sanity 对比，确认 image baseline 不再使用从头训练小 CNN，LiDAR 报告能清晰显示是否超过 majority-class baseline。
- [x] 6.4 如果 LiDAR 仍未超过 majority-class baseline，基于质量诊断记录下一步 ablation 候选：FoV 收窄、ROI 调整、BEV cache 重建或新增 PointNet-style raw point encoder。

## 7. 文档与清理

- [x] 7.1 更新 README、训练说明或相关中文方案文档，说明默认 camera encoder 已切换为 ImageNet 预训练 ResNet-18。
- [x] 7.2 更新 LiDAR 方案文档，说明默认 baseline profile、normalizer/cache 使用方式和退化基线解释。
- [x] 7.3 检查 `openspec validate` 或等价命令，确保 change artifacts 和 delta specs 可通过 OpenSpec 校验。
