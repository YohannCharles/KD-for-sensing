## Context

上一轮 `optimize-multimodal-nf-throughput` 已经为 Multimodal-NF image/LiDAR 建立了派生 `.npy` cache，避免训练时重复解压原始 gzip HDF5。当前实测显示 GPS-only 训练可以快速完成，但 image/LiDAR/fusion 在 6 个后台程序并行时仍然慢：GPU 利用率波动，DataLoader worker 多次进入不可中断 IO 等待，系统 IO wait 明显升高。

新的瓶颈有两层：

1. dataset 初始化阶段：`read_only` cache 解析仍会为每个原始 HDF5 计算 SHA256 fingerprint。多个训练同时启动时，会重复扫描原始大文件。
2. 训练阶段：每个 worker 通过 `np.load(..., mmap_mode="r")` 读取按 city 存储的 100GB 级派生 cache。epoch 子采样默认随机打散样本，多个进程同时随机访问 image/LiDAR 大文件，page cache 难以命中，DataLoader wait 出现尖峰。

本 change 需要在不改变样本字段、target 语义、模型接口和指标口径的前提下，让 Multimodal-NF 的 cache 读取路径可诊断、可控制，并给后台并行训练更安全的推荐。

## Goals / Non-Goals

**Goals:**

- 让 `read_only` 或 warm cache 的 dataset 初始化避免重复扫描原始 HDF5 大文件。
- 为 Multimodal-NF 派生 cache sidecar 增加轻量一致性字段、强校验状态和 IO 布局元数据。
- 提供局部性优先的 train 子采样/采样顺序选项，使随机抽样仍可复现，但 batch 读取尽量按 dataset/source 顺序聚集。
- 扩展 profile 输出，使用户能看到 cache 校验、cache open/read、DataLoader wait 和 IO-risk 判定。
- 扩展并行训练推荐器，对 Multimodal-NF image/LiDAR/fusion 输出 cache 校验模式、采样局部性、AMP、progress、worker 和 GPU 分配建议。
- 用 fixture 和 focused tests 覆盖行为，不依赖真实全量数据或提交 cache 产物。

**Non-Goals:**

- 不改变 Multimodal-NF target、codebook flatten、LOS/link 语义或训练指标解释。
- 不把真实数据、派生 cache、profile 输出、日志或 checkpoint 纳入源码。
- 不强制所有训练默认启用局部性采样；旧配置必须继续可运行。
- 不引入需要服务端进程的新外部存储系统。
- 不把模型结构优化作为本 change 的必需部分。

## Decisions

### 1. 将运行时 cache 校验从“强 fingerprint”改为“轻量默认 + 显式强校验”

派生 cache sidecar 继续记录 `source_fingerprint`，但 dataset 初始化默认不重新计算原始 HDF5 SHA256。运行时默认使用轻量 identity 校验，例如原始路径、文件大小、mtime ns、profile、split、seq_len、num_pred、shape、dtype、cache version 和生成时记录的 fingerprint 字段是否存在。若用户显式设置强校验或运行预处理校验命令，才重新计算原始文件 fingerprint。

理由：训练启动时重复扫描原始 HDF5 不能提升每次训练的样本正确性，却会在多进程启动时制造可观 IO。强校验仍然保留给 cache 生成、审计或疑似数据漂移场景。

备选方案：完全删除 fingerprint。拒绝，因为这会降低 cache 与原始数据关系的可审计性。

### 2. 保持派生 cache 为本地文件产物，但补充 IO 布局元数据和读取边界

短期不强制替换现有 `.npy` 格式；先让 sidecar 记录 `storage_kind`、`layout`、`shard_key`、`sample_count`、`bytes`、`recommended_access_pattern`、`validation_mode` 和 `validation_duration_seconds`。dataset adapter 继续 lazy open cache 文件，但 runtime metadata/profile 必须能暴露实际打开的 cache 文件数、映射字节数和读取耗时。

如果实现中增加新的 shard 或 window cache 格式，必须保持与原始 sample 等价，并通过 sidecar 记录格式版本和回退方式。推荐先优先解决随机访问局部性，再评估是否需要更大磁盘成本的 window materialization。

理由：现有派生 cache 已经可用，直接改成全量 window cache 会将 image/LiDAR 磁盘占用放大约 `seq_len` 倍，风险太高。

备选方案：一次性转换为 Zarr/LMDB 等新格式。暂不采用，除非 profile 证明局部性采样仍无法满足吞吐。

### 3. 用局部性优先采样解决大 cache 随机读

扩展 `training.epoch_subsampling` 或 DataLoader sampler，使其支持“随机选择样本，但按局部性排序/分块输出”的策略。建议语义：

- `shuffle=true`: 保持现有完全随机顺序。
- `shuffle=false`: 现有行为，随机选子集后排序，可作为最小可用局部性路径。
- 新增可选 `order` 或 `locality` 字段：例如 `random`、`sorted`、`source_block`、`block_shuffle`。实现时可先让推荐器输出 `shuffle=false`，再逐步增加 block shuffle。

理由：Multimodal-NF 的 cache 文件按 city/source 存储，dataset index 顺序天然更接近顺序读。随机子集后排序能保持覆盖随机性，同时显著减少跨文件 page fault。

备选方案：只减少 DataLoader worker。该方案能降低并发 IO，但不能解决单 run 内随机读带来的尾延迟。

### 4. profile 必须把“慢在哪里”结构化输出

`scripts/profile_training_io.py` 应在 Multimodal-NF 下记录：

- dataset 初始化耗时和 cache validation 耗时；
- 每个模态 cache source kind、validation mode、cache path 数、cache 总字节；
- cache open/read 计数、均值、P95 和最大值；
- DataLoader wait、transfer、forward/backward、samples/s；
- IO-risk summary，例如 `cache_random_read_risk`、`loader_wait_dominates_step`、`cache_validation_scan_detected`；
- AMP、progress、train/test worker、prefetch、pin memory、epoch_subsampling 配置。

理由：用户需要从一次 profile 中判断是 cache 初始化、随机读、worker 配置、GPU 计算还是日志输出拖慢，而不是依赖截图猜测。

### 5. 并行推荐器对 Multimodal-NF 使用单独策略

`scripts/recommend_parallel_training.py` 对 Multimodal-NF image/LiDAR/fusion 应输出更明确的建议：

- cache warm 后优先 `read_only` + 轻量校验，不建议每个训练重复强校验；
- 启用 `training.epoch_subsampling.shuffle=false` 或后续局部性字段；
- 后台训练默认 `output.progress.enabled=false`；
- CUDA image/fusion 推荐 AMP；
- 对 image/LiDAR/fusion 重 IO run，建议先按 GPU 均匀分配，避免同一 GPU 上叠多个重 IO run 后互相等待；
- 根据 parallel runs、CPU 数和 cache IO-risk 给出 worker 上限，而不是只按 CPU 核数放大 worker。

理由：MMW、scenario9、scenario32 的并行经验不能直接迁移到 Multimodal-NF image/LiDAR，因为后者的主要瓶颈是本地大 cache 随机读。

## Risks / Trade-offs

- 轻量校验可能漏掉原始 HDF5 内容被原地改写但大小/mtime 未变的极端情况 → sidecar 保留生成时 strong fingerprint，并提供显式强校验/重建命令；强校验结果写入 metadata。
- 局部性排序会改变每 epoch batch 顺序 → 保持样本选择可复现，并在 run metadata 中记录排序策略；默认旧行为兼容。
- `shuffle=false` 可能降低随机 batch 混合度 → 推荐用于 IO 受限的后台并行训练；后续可用 block shuffle 在随机性和局部性之间折中。
- profile 增加 instrumentation 可能有少量开销 → 只在 profile 脚本或显式诊断模式中采集细粒度耗时；训练默认只记录轻量 metadata。
- 新 cache layout 元数据与旧 sidecar 不兼容 → 读取旧 sidecar 时按缺省值补齐；需要强校验或重建时输出清晰提示。

## Migration Plan

1. 为派生 cache sidecar 和读取计划增加轻量 validation metadata；默认运行时不重算原始 HDF5 SHA256。
2. 增加显式强校验/重建入口，并在预处理输出中记录 validation mode、耗时和结果。
3. 扩展 dataset adapter 和 profile instrumentation，记录 cache open/read 与 IO-risk 字段。
4. 扩展 epoch 子采样或推荐器，先输出 `training.epoch_subsampling.shuffle=false`，必要时实现 block/source locality sampler。
5. 更新 Multimodal-NF 推荐器和示例配置，覆盖 AMP、progress、worker、cache validation 和 GPU 分配建议。
6. 用 focused tests 和一次小样本 profile 验证行为；真实全量数据上的吞吐对比作为本地验证产物，不提交。

回退策略：将 cache policy 改为 `off` 使用原始 HDF5，或将采样策略恢复为 `shuffle=true`；删除 ignored 派生 cache 不影响原始数据。

## Open Questions

- 是否需要在首个实现中新增 `training.epoch_subsampling.order`，还是先只让推荐器使用现有 `shuffle=false`？
- 是否需要实现 cache mmap LRU 上限，限制每个 worker 同时打开的 `.npy` 文件数？
- 如果局部性采样后仍慢，下一步 cache 格式应优先评估 per-source 小 shard、row-group HDF5，还是 Zarr/LMDB？
