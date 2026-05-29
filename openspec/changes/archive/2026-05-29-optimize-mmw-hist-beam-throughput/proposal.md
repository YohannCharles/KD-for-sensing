## Why

MMW Town10 的 HiST-Beam 多模态 LOSO 训练当前受 CPU 侧 image 序列解码、DataLoader worker 内存膨胀和重复 source/prototype 数据扫描限制，表现为 GPU 显存占用低、系统内存占用高、单 epoch 可达十几分钟甚至被 OOM killer 终止。这个问题已经阻塞完整 MMW radio-semantic 实验矩阵的可复现执行，需要把临时命令行降载升级为可配置、可诊断、可回归的长期机制。

## What Changes

- 为 MMW RGB/ImageNet image 序列增加受控派生缓存能力，支持预热、按需生成、只读复用和 profile/report 记录，避免每个 epoch 对同一批 PNG 反复解码与 resize。
- 调整 MMW/HiST-Beam LOSO 数据构建与执行边界，使 source training 阶段默认只构建当前 stage 必需的数据集和 DataLoader，target adapt/test 数据集延后到对应 stage 构建。
- 为 HiST-Beam source prototype 生成增加策略开关、进度诊断和 variant-aware 跳过规则，避免不需要 prototype 的 source-only baseline 在 source training 后再次完整扫描 image-heavy source split。
- 扩展训练吞吐 profile 与并行推荐，识别 MMW image 序列、`seq_len`、batch size、worker 数和并行 run 数导致的 CPU 内存风险，并输出更保守的覆盖建议。
- 收紧 MMW dataset 初始化和 normalizer 行为，避免为了 GPS/mmWave normalization 在 train dataset 初始化阶段无界缓存全部样本特征；运行产物需要记录 normalizer/cache 的来源与内存相关配置。
- 不引入旧 image motion cache，也不恢复已删除的 motion mask 路径；新的 image cache 仅针对 RGB/ImageNet 输入或 image encoder feature，且必须保持样本契约一致。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `training-throughput-optimization`: 增加 MMW/HiST-Beam image-heavy 训练的 profile 字段、内存风险判定、并行推荐和回归验证要求。
- `automated-cache-policy`: 将受支持 cache 扩展到 MMW RGB/ImageNet image 派生缓存，并保持 image motion cache 被拒绝的契约。
- `modality-aware-data-loading`: 约束 MMW dataset 的按需加载、normalizer 拟合和大数组缓存边界，避免 dataset 初始化或 worker 复制导致无界 CPU 内存增长。
- `hist-beam-cross-scene-adaptation`: 调整 LOSO stage 数据构建边界和 source prototype 生成策略，使 prototype 只在需要时生成并带有进度诊断。

## Impact

- 影响数据加载与缓存：`src/kd_sensing/data/datasets/*`、`src/kd_sensing/data/transform_ops/image.py`、cache policy 与预处理入口。
- 影响 LOSO 执行器：`src/kd_sensing/engine/loso_data.py`、`src/kd_sensing/engine/hist_beam_loso_execution.py`、prototype 生成逻辑。
- 影响吞吐诊断与推荐：`scripts/profile_training_io.py`、`scripts/recommend_parallel_training.py`、相关 runtime metadata。
- 影响配置与验证：`configs/hist_beam/mmw_scenario_loso.yaml` 的推荐覆盖、focused tests、OpenSpec spec delta 和文档说明。
- 不应改变现有 beam label、image tensor shape、GPS/mmWave feature shape、HiST-Beam 模型输出或 LOSO metric 语义。
