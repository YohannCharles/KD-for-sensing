## Why

Multimodal-NF 的 image/LiDAR 派生缓存已经消除了原始 gzip HDF5 重复解压，但实际并行训练仍然很慢：`read_only` 初始化会重复对原始 HDF5 做全量 fingerprint，训练阶段又对 100GB 级 `.npy` mmap cache 做随机窗口读取，导致 DataLoader worker 进入 IO 等待而 GPU 利用率波动。需要把上一轮吞吐优化从“有缓存可读”推进到“缓存校验、布局、采样和并行推荐都对 IO 友好”。

## What Changes

- 调整 Multimodal-NF image/LiDAR 派生缓存校验契约：运行时读取缓存时 MUST 避免每次构建 dataset 都重复扫描原始大文件；sidecar MUST 提供可审计的轻量一致性字段，并支持需要时显式执行强校验。
- 为 Multimodal-NF 派生缓存增加 IO-aware 读取契约：缓存格式、分片或访问策略 MUST 支持随机窗口训练时稳定读取，避免多进程同时 mmap 大文件后产生不可控 page fault 和磁盘等待。
- 扩展 train epoch 子采样和并行训练推荐：当启用 Multimodal-NF image/LiDAR 且使用大派生缓存时，推荐器 MUST 能提示顺序/局部性优先的子采样、AMP、progress 降噪、worker 数和 GPU 分配建议。
- 扩展训练吞吐 profile：对 Multimodal-NF 输出 cache 校验耗时、cache open/read 耗时、随机读/顺序读信号、DataLoader wait 尖峰和 IO-risk 判定字段。
- 更新 Multimodal-NF 示例配置或推荐输出，使含 image/LiDAR 的后台并行训练默认不会误导用户把 `read_only` cache 当作一定高速路径。
- 增加 focused tests，覆盖轻量 cache 校验、强校验入口、IO-aware metadata/profile 字段、子采样局部性配置和推荐器输出。
- 不改变 Multimodal-NF 样本字段、target 语义、codebook flatten 规则、模型前向接口或指标解释口径。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `multimodal-nf-dataset`: 增加 Multimodal-NF 派生缓存轻量校验、强校验、IO-aware cache metadata 和随机窗口读取约束。
- `training-throughput-optimization`: 增加 Multimodal-NF cache IO profile 字段、IO-risk 判定、并行运行推荐和 train 子采样局部性建议。

## Impact

- 影响 `src/kd_sensing/preprocessing/multimodal_nf_derived_cache.py`、`src/kd_sensing/data/datasets/multimodal_nf.py`、`src/kd_sensing/engine/epoch_subsampling.py`、`src/kd_sensing/engine/throughput_recommendations.py`、`scripts/profile_training_io.py`、`scripts/preprocess.py` 和 Multimodal-NF 配置样例。
- 新增或扩展测试主要位于 `tests/test_multimodal_nf_dataset.py`、`tests/test_training_io_workflow.py`，必要时新增 focused cache IO/recommendation 测试。
- 派生缓存、profile 输出、预热报告、训练日志和 checkpoint 仍属于本地产物，默认位于 `dataset/MultimodalNF/cache`、`outputs/` 或用户显式 ignored 目录，不纳入源码变更。
- 不引入旧入口、兼容聚合层或绕过 `src/kd_sensing` 包结构的运行方式。
