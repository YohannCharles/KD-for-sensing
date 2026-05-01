## Why

当前三模态和包含 image 的 fusion 训练出现 GPU 利用率间歇性升高、大多数时间接近 0% 的现象，更像训练输入管线饥饿，而不是模型显存不足。Scenario 9 的滑动窗口高度重叠，训练 split 中 image、radar、GPS、LiDAR 帧引用重复因子约 6.6x；其中 image motion mask 每个样本都会重复读取 8 张 jpg、resize、灰度化、Gaussian 滤波和差分，是当前 CPU 路径最重的部分。LiDAR BEV cache 虽已具备参数 hash 目录，但本地 cache 可能为空，导致训练仍在线从 `.mat` 构造 BEV。

需要把“训练前/训练中喂数据”的路径变成可观测、可缓存、可复用且配置上适合并行实验的工作流，然后再启用 non-blocking GPU transfer、AMP 和 batch size 调优来提高吞吐。

## What Changes

- 新增训练吞吐 profiling/benchmark 脚本，分别记录 dataset `__getitem__`、DataLoader 取 batch、CPU 到 GPU transfer、forward/backward 和整体 step 时间。
- 新增 image motion mask 预处理缓存，按相邻帧 pair 或等价稳定 cache key 保存 `uint8`/bool mask，并以预处理参数 hash 子目录和 `metadata.json` 隔离不同参数。
- 扩展 Scenario 9 dataset，使 image motion mask 可优先从缓存读取，缺失时可按配置在线生成并写入，且不改变既有返回张量契约。
- 补齐 LiDAR BEV cache 预热工作流，支持 train/test CSV、跳过已存在缓存、显示进度、写出 sidecar metadata，并保证训练配置可直接复用。
- 缓存 beam label 的 `np.loadtxt + argmax` 结果，避免每个样本重复解析同一个 beam 文本；优先在 Dataset 初始化阶段建立轻量 path-to-label 字典。
- 调整并行实验友好的 DataLoader 推荐配置和 canonical YAML，减少多个实验同时运行时 `num_workers * prefetch_factor` 放大的 CPU/I/O 压力。
- 在 batch 准备中支持 pinned-memory 场景下的 `non_blocking=True` 传输，并在训练配置中加入 AMP 开关、`GradScaler` 和可测试的 FP32 回退。

## Capabilities

### New Capabilities
- `training-throughput-optimization`: 覆盖吞吐 profiling、image motion mask cache、beam label cache、non-blocking transfer、AMP 和吞吐相关运行记录。

### Modified Capabilities
- `lidar-preprocessing`: 扩展 BEV cache 预处理入口，要求 train/test cache 可预热、可跳过已存在文件、可记录参数 metadata。
- `modality-aware-data-loading`: 增加 image motion mask cache 与 beam label cache 的懒加载/复用要求。
- `experiment-workflow`: 增加 AMP、DataLoader 并行实验默认建议和吞吐 profile 输出记录。

## Impact

- 主要影响 `src/kd_sensing/data/transforms.py`、`src/kd_sensing/data/datasets/scenario9.py`、`src/kd_sensing/preprocessing/lidar.py`、`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/config/defaults.py` 和相关 YAML。
- 需要新增 image motion mask 预处理配置与脚本入口，并复用现有 preprocessing registry 或 CLI。
- 需要补充单元测试覆盖 cache key、metadata、cache hit/miss、参数变化自动分流、beam label 缓存和 non-blocking/AMP 配置解析。
- 需要用 `conda run -n kd_mm_beam ...` 运行目标测试和短训练/profile smoke test。
