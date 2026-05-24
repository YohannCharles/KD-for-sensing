## Context

当前项目已经有中心化模态契约、配置驱动 dataset factory、flat batch runtime 和 registry，但 dataset 实现还主要以具体数据集为中心：DeepSense6G 负责 CSV 序列、future beam、GPS/mmWave/CSI/LiDAR scaler/cache；MMW 继承 DeepSense6G 并在初始化阶段补 CSV 列；Raymobtime s008 是 NPZ cache snapshot。它们对训练侧暴露的字段大体统一，但内部样本索引、target、metadata、cache 和 artifact 传递方式不统一。

Multimodal-NF 来自 `https://lmyxxn.github.io/6GXLMIMODatasets/`、`https://github.com/Lmyxxn/Multimodal-NF` 和 Hugging Face `lmyxxn/MultimodalNF`。官方信息显示它面向低空 UAV 近场 XL-MIMO，包含同步近场 CSI、RGB、LiDAR 点云、GPS/trajectory 信息和 wireless labels；默认配置包含 64x64 UPA、7 GHz、UAV 高度 5-80 m、每轨迹 20 帧、0.1 s 采样，并按城市划分 train/val/test。数据侧包含 HDF5 channel/image/lidar 文件、Top-5 三维 beam codebook 索引、beam power、LoS/NLoS、NF/FF、轨迹和 city metadata；codebook 提供 dense `90 x 45 x 16` 和 small `20 x 20 x 10` 两种。

这意味着 Multimodal-NF 不是 DeepSense6G 的一个新 scene，也不是 Raymobtime s008 的同构 snapshot cache。它是一个新的 dataset family，同时会触发项目里“dataset 统一抽象”的需求。

## Goals / Non-Goals

**Goals:**

- 以 `multimodal_nf` dataset type 接入 Multimodal-NF，本地默认目录为 `dataset/MultimodalNF/`，允许显式 `data_root`。
- 支持官方 HDF5/codebook 文件的审计、索引、轻量 cache 和 split metadata，不把真实数据或生成产物提交进源码。
- 定义 dataset runtime contract，使后续新增 UAV/空间 codebook 数据集只需要实现 descriptor、sample index、modality adapter 和 target provider。
- 保持现有 DeepSense6G、MMW、Raymobtime 输出字段和配置兼容；迁移是渐进式，不一次性重写现有 dataset。
- 支持 Multimodal-NF 当前 frame near-field beam selection，保留 Top-5 三维 codebook 候选和 beam power metadata。
- 支持 RGB、LiDAR 点云、UAV 3D 位置、near-field CSI 的单模态和 fusion 数据加载。
- 所有 Python 预处理、测试、训练和验证命令使用 `conda run -n kd_mm_beam ...`。

**Non-Goals:**

- 不在本变更中下载 Hugging Face 数据或提交任何 `.h5`、`.zip`、`.pkl` codebook、cache、checkpoint、日志。
- 不把 Multimodal-NF 映射成 DeepSense6G future beam prediction，也不默认使用 DeepSense6G 的 `seq_len=8,num_pred=3` 语义。
- 不在第一阶段训练完整大模型或要求真实全量数据可用；首版以 contract、fixture、smoke workflow 和可扩展接口为准。
- 不删除现有 DeepSense6G/MMW/Raymobtime dataset 类；只在新路径稳定后逐步迁移。
- 不把 `gps` 和 `csi` 改名为新模态；通过 profile 表达 dataset-specific 输入语义，避免破坏已有配置。

## Decisions

### 1. 使用组合式 runtime contract，而不是厚基类继承

新增 `kd_sensing.data.dataset_runtime` 或等价窄模块，提供以下概念：

- `DatasetDescriptor`: 数据集家族、默认路径、存储类型、split 语义、支持模态、默认 target schema 和 metadata。
- `SampleIndex`: 把 CSV、HDF5、NPZ cache 或 manifest 统一成轻量 rows；row 只保存 `sample_id`、split、scene/city、trajectory、frame、资源引用和 target 引用。
- `ModalityAdapter`: 每个模态/profile 负责字段校验、懒加载、shape/dtype 标准化、cache key 和 normalizer artifact。
- `TargetProvider`: 每个 objective/target schema 负责 label、valid mask、辅助 target 和 target metadata。
- `RuntimeDataset`: 薄 PyTorch Dataset，只组合 index、adapters 和 provider，生成 flat sample dict。

替代方案是建立一个 `BaseDataset`，要求每个数据集继承并覆盖大量 `load_*` 方法。这个方案短期看简单，但 Raymobtime snapshot、DeepSense future sequence、Multimodal-NF frame-wise 3D codebook 的语义差异太大，会很快退化成空方法和特判，因此不采用。

### 2. Multimodal-NF 作为新 dataset family，不复用 DeepSense6G scene 机制

新增 layout descriptor：`dataset/MultimodalNF/`。建议目录语义为：

```text
dataset/MultimodalNF/
  raw/          # 用户放置官方 HDF5/zip 原始数据，可通过配置改名或外部路径覆盖
  codebooks/    # 用户放置 upa64x64_NF_codebook*.pkl
  cache/        # 可生成 index/cache，本地产物
```

审计工具需要支持两类来源：

- Hugging Face 文件：`MultiSubcarrier/City_*.h5`、image/lidar zip、codebook `.pkl`。
- 项目页描述的分目录 HDF5：channel dataset、image h5、lidar h5 按 `City_###` 对齐。

第一版不假设用户已经解压所有 zip。审计阶段输出实际发现的文件、HDF5 keys、shape、dtype、city 列表和 codebook 文件；cache builder 根据显式配置或审计结果选择可用来源。

### 3. 通过模态 profile 扩展 `gps`、`csi`、`lidar`，减少新模态数量

Multimodal-NF 的 `Pos[N,3]` 是 UAV 当前 3D 坐标，和 DeepSense6G 的历史 relative-polar GPS 不同；`H` 是近场 XL-MIMO channel tensor，和现有 CSI 降级实验也不同；LiDAR 是 10,000 点云，不是 DeepSense LiDAR BEV，也不是 Raymobtime occupancy grid。因此使用 profile 表达输入契约：

- `gps_profile: uav_xyz_snapshot`，sample key 仍为 `gps`，形状 `[1, 3]` 或 batch 后 `[B, 1, 3]`。
- `csi_profile: xl_mimo_nf`，sample key 仍为 `csi`，支持窄带 `[1, 4096, 1, 2]` 和多子载波 `[1, 4096, K, 2]`。
- `lidar_profile: point_cloud_xyz_10000`，sample key 仍为 `lidar`，形状 `[1, P, 3]`，P 默认 10000。
- `image_profile: rgb_imagenet` 继续复用已有 RGB/ImageNet 契约，dataset 或 runtime 负责从 HDF5 RGB 转成 `[1, 3, H, W]`。

替代方案是新增 `uav_position`、`nf_channel`、`point_lidar` 三个模态。这样会扩大模型、batch helper、诊断和配置矩阵，且和已有 `gps/csi/lidar` 概念重复，所以第一版不采用。

### 4. Near-field beam target 保留三维 codebook 信息，同时输出兼容 `target_beam`

Multimodal-NF 的 `BeamIdx` 是 Top-5 三维 triplet，例如 `[azimuth_idx, elevation_idx, range_idx]`。训练主目标需要兼容现有分类流程，因此 dataset 返回：

- `target_beam`: Top-1 triplet 按 codebook shape flatten 后的 class id，形状 `[1]`。
- `beam_triplet_topk`: Top-5 triplet，形状 `[5, 3]`。
- `beam_power_topk`: Top-5 beam power，形状 `[5]`。
- `beam_codebook_shape`: 通过 metadata 或 dataset 属性记录，例如 `[90, 45, 16]` 或 `[20, 20, 10]`。
- `los_label`、`nf_label`、`trajectory_mode` 等作为辅助 target 或 metadata。

新增 `near_field_beam_selection` objective。它默认使用 `target_beam` 做 CE/Focal loss，指标包含 `val_beam_top1/top3/top5`，并可在 metadata 中记录 triplet Top-K 命中和 codebook shape。这样既能跑现有分类模型，又不丢失 3D codebook 结构。

### 5. HDF5 加载必须懒加载，禁止初始化时物化大数组

Multimodal-NF channel 数据体量大，尤其 `H[N,4096,K,2]` 和 image/lidar HDF5/zip。dataset 初始化只读取 index 和 metadata，不把全量 H/image/lidar 加载到内存。多 worker DataLoader 下，每个 worker 延迟打开自己的 HDF5 handle，并在 worker 生命周期内复用。

配置提供轻量选项：

- `csi_subcarrier_policy`: `single`、`all`、`select`。
- `city_subset`、`portion`、`split_mode`。
- `materialize_cache`: 是否预先写轻量 index/cache，默认只写 index，不写大 tensor cache。

### 6. Data factory 由 descriptor 决定 split 解析

当前 `build_dataset` 对非 synthetic dataset 默认套 `train_csv_name/test_csv_name`。新增 descriptor 后，factory 根据 `storage_kind` 处理：

- `csv_sequence`: 保持 DeepSense6G/MMW 现有 CSV 行为。
- `npz_snapshot`: 保持 Raymobtime cache 行为。
- `hdf5_frame`: Multimodal-NF 使用 descriptor/index 解析 split，不要求 `csv_name`。

这能解决“新增数据集必须伪装成 CSV”的问题，也避免后续 UAV 数据集继续在 dataset `__init__` 中绕过 factory。

### 7. 迁移路线按“先新增、后收敛”执行

阶段 A：新增 runtime contract 和 Multimodal-NF，小 fixture 跑通审计、index、dataset、objective。现有 dataset 不迁移。

阶段 B：把 DeepSense6G 的 `create_samples`、beam target 和 modality loader 包一层 descriptor/index/adapter façade，但保持 `DeepSense6GDataset` 公共类和输出字段不变。

阶段 C：把 MMW 的动态 CSV 补列逻辑迁到 preparation/index builder，`MMWDataset` 只消费规范 index 或继承已收敛的 DeepSense sequence descriptor。

阶段 D：Raymobtime s008 保留 snapshot 语义，但使用同一 target/artifact metadata 接口；不改变其 current selection objective。

阶段 E：当三个真实数据集都通过 runtime contract 后，再清理重复的 dataset 内部 helper。任何清理都必须另开 OpenSpec change。

### 8. Multimodal-NF fusion 默认要约束 CPU 并缓存 HDF5 key 解析

真实三模态 fusion 同时读取 image、LiDAR 点云和 GPS。若训练进程不显式设置 `training.cpu_threads`，PyTorch 会沿用默认 CPU 线程池；在多实验并行时，单个 fusion 主进程可能把几十个 CPU 线程拉满，表现为 CPU 占用远高于三个单模态的线性相加。同时，早期 Multimodal-NF index 只记录 channel HDF5 的 key，未记录独立 image/lidar HDF5 key；dataset 因此会在取样时按样本重复遍历 HDF5 dataset 路径。

决策：

- Multimodal-NF 三模态 fusion 示例配置显式设置较保守的 `training.cpu_threads`，并降低 train worker/prefetch 与 test worker，作为多实验并行时的默认安全值。
- worker-local HDF5 adapter 缓存每个文件的 dataset path/key 解析结果，使旧 index 即使缺少 image/lidar key，也只在每个 worker 首次访问文件时遍历一次 HDF5 结构。
- Multimodal-NF dataset 支持 split-specific `eval_portion`，常规调参配置可只抽样 val/test split 做每轮指标，训练 split 默认仍使用完整样本；最终报告或复现实验可覆盖回 `eval_portion=1.0`。
- history/target frame index 若是连续递增窗口，HDF5 读取使用 slice 而不是 fancy indexing，保持输出顺序不变并降低每样本取样开销。
- profiling 入口复用训练线程配置，避免 profile 结果和实际训练运行时不一致。

## Risks / Trade-offs

- [Risk] 官方页面、Hugging Face 和生成器 repo 的文件布局可能不完全一致。→ 审计工具必须先打印实际文件和 HDF5 keys/shape，cache builder 支持 `source_layout: auto`，错误信息包含实际发现内容。
- [Risk] HDF5 多 worker 文件句柄处理不当会导致崩溃或性能下降。→ dataset 不在主进程持久化打开句柄；worker 内懒加载并可显式关闭；单测覆盖 `num_workers=0`，smoke 可覆盖多 worker。
- [Risk] `gps/csi/lidar` profile 扩展可能影响旧配置。→ profile 默认为现有 DeepSense/Raymobtime 行为；旧配置不设置 Multimodal-NF profile 时行为不变；架构边界测试覆盖旧配置。
- [Risk] 3D codebook flatten 后丢失空间结构。→ 主 loss 先使用 flat class 保持兼容，同时保留 `beam_triplet_topk`、`beam_power_topk` 和 codebook shape，用于后续结构化 loss/metric。
- [Risk] LiDAR 点云模型和现有 BEV/occupancy 模型不兼容。→ 第一版 dataset contract 返回 point cloud profile；模型配置必须显式选择支持 point cloud 的 encoder，或通过 adapter 转换成 BEV/voxel。
- [Risk] 全量 Multimodal-NF 数据很大，完整训练不适合作为 CI。→ 使用小型 HDF5 fixture、portion smoke 和 help/contract 测试作为验证主体。

## Migration Plan

1. 创建 OpenSpec 契约和小型 fixture，固定 Multimodal-NF schema、runtime contract 和迁移边界。
2. 实现 layout descriptor、审计工具、HDF5 index builder 和 codebook metadata 解析。
3. 实现 runtime contract 的最小可用版本，并用 Multimodal-NF dataset 首先消费该 contract。
4. 接入 `near_field_beam_selection` objective、batch helper profile 和 smoke 配置。
5. 运行 focused tests 和 OpenSpec 校验：`openspec validate add-multimodal-nf-dataset --strict`、`conda run -n kd_mm_beam pytest ... -q`。
6. 在后续 change 中逐步迁移 DeepSense6G/MMW/Raymobtime 内部实现，保持公共 dataset type 和输出字段兼容。

Rollback 策略：该 change 默认新增 `multimodal_nf` 和 runtime contract，不改变旧配置默认路径；若真实数据布局阻塞，可保留 descriptor 和审计工具，暂时禁用 `multimodal_nf` 训练配置，不影响现有工作流。

## Open Questions

- 官方 release 中 image/lidar zip 与 channel HDF5 的 city/trajectory/frame 对齐键是否完全一致，需要用真实文件审计确认。
- `H` 默认训练输入应使用窄带 `K=1`、固定子载波选择，还是支持多子载波全量输入，需要结合模型和显存预算决定。
- `BeamIdx` triplet 的维度顺序是否固定为 `[azimuth, elevation, range]`，以及 dense/small codebook 是否共用同一 flatten 顺序，需要用 codebook 文件确认。
- LiDAR 首版模型路径是直接 point cloud encoder，还是 adapter 先转 BEV/voxel，需要根据现有模型复用成本和实验目标决定。
