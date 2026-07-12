## Why

已下载的 MMW rainy 和 foggy 场景尚未准备，sunny 产物仍使用旧的 8 帧输入、3 帧预测窗口，且现有 LMDB 生成器硬编码为 DeepSense6G。当前实验主线统一为输入窗口 5、预测窗口 1，需要让全部已下载 MMW 条件具有一致、可复现的准备和缓存产物。

## What Changes

- 为 sunny、rainy、foggy 下全部已下载 Town03 场景生成 H5/P1 sequence split。
- 让 split-level LMDB 样本缓存生成器可通过当前 dataset registry 构建 DeepSense6G 或 MMW dataset，同时保留现有 DeepSense6G 入口。
- 为 MMW 三种天气提供可复用的 H5/P1 图像、LiDAR 和 LMDB 预处理配置，缓存默认位于 `outputs/cache/MMW/<condition>/`。
- 增加缓存 metadata 和 H5/P1 窗口契约的聚焦测试。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `mmw-town10-dataset-preparation`: 扩展已下载 MMW 多天气 H5/P1 准备和 split-level LMDB 缓存要求。

## Impact

- 影响 `src/kd_sensing/preprocessing/sample_cache.py`、预处理 CLI action、MMW 预处理配置和相关测试。
- 本地生成的 dataset、LMDB、图像/LiDAR cache 和报告继续属于 ignored runtime artifacts，不进入源码提交。
