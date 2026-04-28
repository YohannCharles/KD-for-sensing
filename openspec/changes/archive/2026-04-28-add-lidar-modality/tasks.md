## 1. 数据与预处理

- [x] 1.1 扩展 `SequenceSamples` 和 `create_samples`，支持解析可选 `lidar1..lidarN` 列，并保持旧 CSV 兼容。
- [x] 1.2 扩展 `generate_sequence_data` / `SequencePreprocessor`，新增 `include_lidar`、LiDAR 源列名和 fallback 配置，能生成带 LiDAR 路径列的 train/test 序列 CSV。
- [x] 1.3 在 `src/kd_sensing/data/transforms.py` 实现 LiDAR 点云读取、无效点过滤、ROI/FoV 裁剪、可选地面过滤和 BEV 三通道构造工具。
- [x] 1.4 实现 LiDAR BEV normalizer，只使用训练 split 统计量 fit，并支持测试 split 复用。
- [x] 1.5 新增 LiDAR BEV 缓存预处理器，支持通过 `conda run -n kd_mm_beam python scripts/preprocess.py --config <config>` 离线生成 `.npy` 缓存。
- [x] 1.6 扩展 `Scenario9Dataset`，新增 `use_lidar`、BEV 参数、缓存参数、normalizer 参数，并在启用 LiDAR 时返回 `lidar` 张量。
- [x] 1.7 扩展 dataloader builder，根据 `experiment.task: lidar` 或 fusion `modalities` 自动启用 LiDAR，并把 train-fitted normalizer 传给 test dataset。

## 2. 模型、Batch 与 Fusion

- [x] 2.1 新增 `src/kd_sensing/models/lidar.py`，实现 `LidarFeatureExtractor`、`LidarModalityNet` 和 `LidarStudentModalityNet`。
- [x] 2.2 注册 `lidar_feature_extractor`、`lidar_teacher`、`lidar_student`，并在 `src/kd_sensing/models/__init__.py` 导出公共类。
- [x] 2.3 在 `src/kd_sensing/engine/batch.py` 实现 `prepare_lidar_inputs`，并让 `forward_model` 支持 `experiment.task: lidar`。
- [x] 2.4 扩展 `prepare_fusion_inputs`，当 `modalities` 包含 `lidar` 时构造 `lidar_batch`，未启用时不要求 batch 存在 `lidar`。
- [x] 2.5 扩展 fusion teacher：允许 `VALID_FUSION_MODALITIES` 包含 `lidar`，并使用 `LidarFeatureExtractor` 构建 LiDAR 分支。
- [x] 2.6 扩展 fusion student：新增轻量 LiDAR CNN/adaptive pooling 分支，并把 LiDAR embedding 纳入 fusion projection。
- [x] 2.7 更新 fusion forward 签名和校验逻辑，支持 `lidar_batch`，并保持旧 image/radar/GPS 调用兼容。

## 3. 配置与文档

- [x] 3.1 更新 `src/kd_sensing/config/defaults.py`，加入 LiDAR dataset 默认参数和 `lidar` task 兼容字段，旧默认 image 配置保持不变。
- [x] 3.2 新增 `configs/lidar/no_kd.yaml`、`configs/lidar/student_no_kd.yaml`、`configs/lidar/logits_kd.yaml`、`configs/lidar/rkd.yaml`。
- [x] 3.3 新增包含 LiDAR 的 fusion 示例配置，包括 all-modalities 和至少一个 LiDAR 双模态配置。
- [x] 3.4 新增 `configs/preprocess` 下的 LiDAR sequence CSV 和 BEV cache 预处理配置示例。
- [x] 3.5 更新 README 或 `docs/extension_guide.md`，说明 LiDAR 数据列、BEV 参数、训练命令和默认不支持二进制 PCD 的限制。

## 4. 测试与验证

- [x] 4.1 新增小型 LiDAR fixture 点云和单元测试，覆盖点云读取、无效点过滤、ROI/FoV 裁剪、空点云和 BEV shape/dtype。
- [x] 4.2 新增 dataset 测试，验证启用 LiDAR 时返回 `[seq_len, 3, H, W]`，未启用 LiDAR 时旧样本行为不变。
- [x] 4.3 新增模型测试，验证 `LidarFeatureExtractor`、`LidarModalityNet`、`LidarStudentModalityNet` forward 输出 shape 和 GRU 参数校验。
- [x] 4.4 新增 batch/fusion 测试，验证 `prepare_lidar_inputs`、`experiment.task: lidar`、包含 LiDAR 的 fusion `modalities` 和缺失输入错误。
- [x] 4.5 新增配置构建测试，验证 `configs/lidar/*.yaml` 和包含 LiDAR 的 fusion 配置能通过注册表构建。
- [x] 4.6 运行 `conda run -n kd_mm_beam pytest`，修复所有失败测试。
- [x] 4.7 使用小比例数据或 fixture 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/lidar/no_kd.yaml -o training.epochs=1 -o data.dataset.portion=0.01`，验证训练、验证和 checkpoint 核心路径。
