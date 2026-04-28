## Why

当前 LiDAR 配置在 `Scenario9Dataset` 初始化阶段会读取整个训练集 BEV，并通过 `np.concatenate` 计算全局归一化统计量，导致训练尚未进入 tqdm/epoch 循环前就出现数十 GB 内存占用和长时间无输出。参考 `LiDAR模态读取方案.md` 中对公开 DeepSense-6G LiDAR 实现的分析，LiDAR 大数据应按样本懒加载，归一化应使用 BEV 构造期固定规则或流式统计，而不是在 Dataset 初始化时全量拼接。

## What Changes

- 修改 LiDAR dataset 读取语义：Dataset 初始化只解析 CSV、保存路径和轻量配置，不得全量读取 LiDAR 点云、BEV 缓存或拼接训练集 BEV。
- 调整 LiDAR 归一化策略：默认不执行全局 z-score；BEV 构造继续输出已裁剪、限幅或局部归一化后的稳定数值范围。
- 新增可选的 LiDAR 流式统计模式，用一次线性扫描按通道累计 mean/std，并把统计量保存为小型 stats 文件供训练、验证和评估复用。
- 保留 `.npy` BEV cache 能力，但 cache 只作为按样本读取加速手段，不作为 Dataset 初始化阶段全量载入的理由。
- 更新 LiDAR-only 和包含 LiDAR 的 fusion 配置默认值，使默认训练入口能快速进入训练循环并在启用输出时显示 tqdm。
- 增加针对大数据初始化行为、归一化复用、缓存读取和配置覆盖的测试。

## Capabilities

### New Capabilities

- 无

### Modified Capabilities

- `lidar-preprocessing`: 明确 LiDAR 点云/BEV 必须按需读取；训练集归一化必须支持禁用和流式统计，且不得在 Dataset 初始化阶段全量 materialize 训练集 BEV。
- `configurable-multimodal-fusion`: 启用 LiDAR 的 fusion 配置必须沿用新的懒加载和归一化默认语义，避免 fusion 训练入口因 LiDAR 初始化全量读取而长时间无进度。

## Impact

- 影响 `src/kd_sensing/data/datasets/scenario9.py` 中 LiDAR normalizer 准备和样本读取逻辑。
- 影响 `src/kd_sensing/data/transforms.py` 中 `LidarBEVNormalizer` 或新增的流式统计辅助逻辑。
- 影响 `configs/lidar/*.yaml`、包含 LiDAR 的 `configs/fusion/*.yaml`、`configs/preprocess/lidar_bev_cache.yaml` 及 README 中的运行建议。
- 影响 LiDAR 数据集、预处理、配置矩阵和训练 smoke test。
