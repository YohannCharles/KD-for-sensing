## Context

当前 LiDAR-only 和包含 LiDAR 的 fusion 配置会在构建 dataloader 时触发 `Scenario9Dataset._prepare_lidar_normalizer()`。该逻辑把训练 split 的每个序列都转换为 BEV，先放入 Python list，再 `np.concatenate` 后计算通道 mean/std。对 3610 条、每条 8 帧、3 通道、224x224 的训练序列，这会在进入 epoch tqdm 前产生数十 GB RSS，并让用户误以为训练没有启动。

`LiDAR模态读取方案.md` 中引用的公开 DeepSense-6G LiDAR 实现采用的是按样本读取点云、即时转换 BEV、固定规则归一化或简单缩放；训练入口不在 Dataset 初始化阶段全量读取 LiDAR。当前项目已经在 BEV 构造时对 height、intensity 和 density 做局部归一化，具备默认关闭全局 z-score 的基础。

## Goals / Non-Goals

**Goals:**

- 让 LiDAR dataset 初始化保持轻量，只解析 CSV、路径和配置，不读取全训练集点云或 BEV。
- 默认 LiDAR 训练能快速进入训练循环，并在 `output.progress.enabled` 为 true 时显示 tqdm。
- 提供内存有界的流式 LiDAR 统计能力，供需要全局 mean/std 的实验显式启用。
- 保留现有 `.npy` BEV cache 作为按样本读取加速能力。
- 保持已有 LiDAR-only、LiDAR KD 和包含 LiDAR 的 fusion 配置可通过命令行覆盖运行。

**Non-Goals:**

- 不重新设计 LiDAR teacher/student 网络结构。
- 不引入新的点云库或 Open3D 依赖。
- 不改变 beam label、序列长度、BEV 通道含义或模型 forward 契约。
- 不要求一次性重新生成全部 BEV cache；cache 仍是可选优化。

## Decisions

1. 默认禁用全局 LiDAR z-score，依赖 BEV 构造期固定范围输出。

   默认配置应设置 `lidar_normalize: false`，或使用等价的新结构化配置 `lidar_normalization.enabled: false`。理由是当前 BEV 构造已经对 height、intensity 和 density 做归一化/限幅，公开参考实现也主要使用固定规则归一化。相比默认全局 z-score，这能避免训练启动前的全量扫描和内存峰值。

   备选方案是保留默认全局 z-score 但改成流式统计。该方案仍会在训练前进行一次全量 I/O 扫描，用户仍会等待较久，因此不作为默认行为。

2. 保留 `lidar_normalize` bool 作为兼容入口，新增结构化归一化配置作为推荐入口。

   推荐配置形态：

   ```yaml
   data:
     dataset:
       lidar_normalization:
         enabled: false
         mode: none
         stats_path: null
   ```

   兼容规则：

   - `lidar_normalize: false` 等价于 `enabled: false, mode: none`。
   - `lidar_normalize: true` 等价于显式启用 `mode: streaming_stats`，但实现必须使用流式统计，不能恢复全量 `concatenate`。
   - 如果同时存在结构化配置和 legacy bool，结构化配置优先。

   这样可以让旧命令行覆盖继续可用，同时给后续实验一个更清晰的配置面。

3. 流式统计按通道累计 sum/sumsq/count，并可保存 stats 文件。

   实现应提供 `LidarBEVStreamingStats` 或等价 helper，逐样本读取 `[T, C, H, W]` BEV 后按通道累计：

   - `sum_c += x.sum(axis=(0, 2, 3))`
   - `sumsq_c += (x ** 2).sum(axis=(0, 2, 3))`
   - `count += T * H * W`

   最终生成 `mean`、`std`、`count` 和元数据，并保存为 `.npz` 或 `.pt` 小文件。验证/测试 split 只能复用训练 stats，不得重新 fit。

   备选方案是用 `np.concatenate` 后一次性 `mean/std`。该方案实现简单但正是本次问题来源，必须移除。

4. Dataset 初始化不执行任何 LiDAR 全量 materialization。

   `Scenario9Dataset.__init__` 可以校验 LiDAR 列是否存在，可以加载小型 stats 文件或 normalizer 参数，但不得遍历全部样本调用 `_lidar_bev_for_index()`。如果需要 on-demand streaming stats，应由 dataloader builder 或预处理命令在显式配置下调用，并显示独立进度。

5. BEV cache 继续按帧文件路径命中，而不是在 Dataset 中缓存整个训练集。

   当前 `_lidar_bev_cache` 的样本级内存缓存会随访问样本数量增长。实现阶段应评估是否默认关闭该内存缓存，或仅在 `lidar_memory_cache.enabled` 显式开启时使用有界 LRU。磁盘 `.npy` cache 可以继续复用，用于减少点云到 BEV 的重复计算。

## Risks / Trade-offs

- [Risk] 默认关闭全局 z-score 可能改变已有 LiDAR 实验数值分布。-> Mitigation：BEV 构造本身保持固定范围归一化；需要复现实验时可显式启用 `streaming_stats` 或提供 stats 文件。
- [Risk] 流式统计仍需要一次全量 I/O 扫描，启动时间可能较长。-> Mitigation：默认不启用；启用时显示独立进度并保存 stats 文件，后续运行直接复用。
- [Risk] legacy `lidar_normalize: true` 语义从 eager full materialization 变为 streaming stats。-> Mitigation：这是性能和内存修复，不改变统计目标；文档说明兼容映射。
- [Risk] 多进程 DataLoader 与 dataset 内部缓存可能造成重复内存占用。-> Mitigation：避免默认样本级无限内存缓存，优先使用磁盘 BEV cache 或小型 stats 文件。

## Migration Plan

1. 将 LiDAR-only 和包含 LiDAR 的 canonical 配置默认改为禁用全局 z-score，保留 BEV size、ROI 和通道参数。
2. 实现结构化 `lidar_normalization` 解析，并保留 `lidar_normalize` bool 兼容。
3. 移除 Dataset 初始化阶段的全量 BEV list/concatenate 逻辑，改为懒加载和可选流式统计。
4. 更新 README 和 `LiDAR模态读取方案.md` 关联说明，给出默认训练命令、启用 `--no-capture-output` 的 conda 运行建议，以及流式 stats 生成/复用示例。
5. 用小比例真实数据和 synthetic/dummy LiDAR 数据做 smoke test，再运行 LiDAR-only 训练 1 epoch 验证 tqdm 和 GPU 进程按预期出现。
