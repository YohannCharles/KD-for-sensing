## Why

Multimodal-NF 在启用 image 或 LiDAR 时训练吞吐显著低于其它数据集，profile 显示主要耗时来自 gzip HDF5 大数组按滑窗反复解压与大张量转换，而 GPS-only 因只读取小型 `[T, 3]` 数组速度正常。需要为 Multimodal-NF 建立可复用的轻量派生数据路径、明确 profile 诊断和推荐配置，避免用户把该问题误判为模型或 CUDA 异常。

## What Changes

- 为 Multimodal-NF image/LiDAR 增加显式、可审计、默认不提交的派生缓存或重打包数据契约，用于减少训练时对原始 gzip HDF5 大 chunk 的重复解压。
- 扩展 Multimodal-NF dataset 配置，使用户可以选择原始 HDF5、只读派生缓存、自动生成派生缓存或强制重建派生缓存，并在运行 metadata 中记录实际来源。
- 扩展训练吞吐 profile，使其能稳定报告 Multimodal-NF 各模态 `__getitem__`、DataLoader wait、transfer、forward/backward 和 samples/s，并给出 image/lidar/gps/fusion 对比所需字段。
- 更新 Multimodal-NF fusion 示例配置建议：对含 image/LiDAR 的训练启用更合理的 DataLoader worker、pin memory、prefetch、test worker、AMP 和 progress 设置。
- 增加 focused tests，覆盖派生缓存命中/缺失、未启用模态不触发缓存、metadata 记录、profile 输出字段和配置兼容性。
- 不改变 Multimodal-NF 样本字段、target 语义、codebook 类别数、模型前向接口或默认训练结果解释口径。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `multimodal-nf-dataset`: 增加 Multimodal-NF image/LiDAR 派生缓存或重打包数据的配置、运行 metadata 和样本等价契约。
- `training-throughput-optimization`: 增加 Multimodal-NF 专用吞吐 profile、瓶颈判定字段、配置推荐和回归验证要求。

## Impact

- 影响 `src/kd_sensing/data/datasets/multimodal_nf.py`、Multimodal-NF preprocessing helpers、data factory/runtime metadata、`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py` 和 `configs/multimodal_nf/*.yaml`。
- 新增或扩展测试主要位于 `tests/test_multimodal_nf_dataset.py`、`tests/test_training_io_workflow.py`，必要时新增 focused throughput/config 测试。
- 生成的派生缓存、profile JSON/CSV、训练输出和任何重打包 HDF5/NPZ/Zarr 文件都属于本地产物，默认必须位于 `dataset/MultimodalNF/cache`、`outputs/` 或用户显式 ignored 目录，不纳入源码变更。
- 不引入新的训练入口，不新增旧兼容聚合层，不绕过当前 `src/kd_sensing` 包结构。
