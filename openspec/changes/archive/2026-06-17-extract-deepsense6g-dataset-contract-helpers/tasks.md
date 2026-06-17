## 1. Helper 边界梳理

- [x] 1.1 标记 `DeepSense6GDataset` 中可抽出的 pure/near-pure contract helper。
- [x] 1.2 新增 GPS contract、column validation、target source 和 cache path helper 模块。
- [x] 1.3 确认 helper 不导入训练、模型或真实资源读取模块。

## 2. 行为迁移

- [x] 2.1 迁移 GPS feature mode normalization 和 scene calibration/angle offset 解析。
- [x] 2.2 迁移 `beam_target_source`、`gps_bev_xy_source` 等支持值校验。
- [x] 2.3 迁移 required columns/error message builder。
- [x] 2.4 迁移纯路径 cache resolution helper，不改变 cache layout。
- [x] 2.5 更新 `DeepSense6GDataset` 调用 helper，保持样本输出兼容。

## 3. 测试和治理

- [x] 3.1 添加 synthetic tests 覆盖 helper 默认值、非法值、missing columns 和 Table III current target 语义。
- [x] 3.2 更新 hotspot budget 和 inventory 拆分说明。
- [x] 3.3 确认不读取真实 `dataset/`、不写入 `outputs/` 或 checkpoint。

## 4. 验证

- [x] 4.1 运行 `openspec validate extract-deepsense6g-dataset-contract-helpers --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q`。
- [x] 4.3 运行相关 DeepSense6G/BeamBench focused tests。
- [x] 4.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
