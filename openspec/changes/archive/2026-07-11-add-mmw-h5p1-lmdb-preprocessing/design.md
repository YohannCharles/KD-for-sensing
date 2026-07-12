## Context

MMW 三种天气使用相同 Town03 场景和 channel 数据结构。现有准备器已能按 condition/scenario 解压、索引并生成 sequence split，图像与 LiDAR 也已有帧级 cache；缺口是当前 sunny split 仍为 8/3、rainy/foggy 未准备，以及 sample LMDB 生成器直接实例化 `DeepSense6GDataset`。

## Goals / Non-Goals

**Goals:**

- 全部已下载 MMW 场景使用 `seq_len=5`、`pred_len=1`。
- 复用 dataset registry 和现有 `LmdbSampleCache` 生成 MMW split-level 样本缓存。
- 保持旧 `deepsense6g_sample_lmdb_cache` 配置可运行。
- 所有可再生成 cache 写入 `outputs/cache/MMW/<condition>/`。

**Non-Goals:**

- 不改变训练模型、label 定义或 group-safe split 算法。
- 不删除下载 zip、已展开原始数据或任何正在运行的训练产物。
- 不引入新的缓存后端或依赖。

## Decisions

- 将现有生成函数内部的数据集构建改为 `DATASETS.build(dataset)`，而不是增加 MMW 专属复制实现。这样 DeepSense6G 与 MMW 共享 key、metadata 和写入行为。
- 新增 `sample_lmdb_cache` 规范入口，并将旧 `deepsense6g_sample_lmdb_cache` 注册名保留为兼容别名。
- MMW LMDB 按 condition/scenario/split 分目录，文件名携带 `seq5_pred1`，避免与旧窗口产物混淆。
- 准备任务按天气和场景并行，但每个 LMDB 仅由一个 writer 写入；不并发写同一 LMDB environment。

## Risks / Trade-offs

- [LMDB 和帧级缓存占用大量磁盘] → 处理前估算容量，处理期间持续检查剩余空间，保留 zip 和原始数据不删除。
- [并行解压造成 I/O 饱和并影响训练] → 限制场景级并发，不使用 GPU，并避免占满全部 CPU。
- [旧 LMDB 配置依赖原类型名] → 保留旧 registry 名和返回 metadata 兼容字段。
- [不同窗口产物被误用] → split metadata、路径和 LMDB metadata 同时记录 `seq_len=5`、`num_pred=1`。
