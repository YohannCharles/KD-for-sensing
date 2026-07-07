# 训练吞吐与预处理缓存

当前优先优化的是训练输入管线，而不是模型显存。典型症状是 GPU 利用率偶尔升高、大多数时间接近 0%，DataLoader 和 CPU 预处理成为瓶颈。

## Profile 命令

所有命令都使用 `kd_mm_beam` 环境。建议先用小样本 profile，确认瓶颈在哪个阶段，再决定是否调 worker、cache 或 batch size。

```bash
conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile \
  --config configs/image/lightweight.yaml \
  --samples 32 \
  --output outputs/profile/image_io.json

conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile \
  --config configs/radar/lightweight.yaml \
  --samples 32 \
  --output outputs/profile/radar_io.json

conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile \
  --config configs/gps/lightweight.yaml \
  --samples 32 \
  --output outputs/profile/gps_io.json

conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile \
  --config configs/lidar/lightweight.yaml \
  --samples 32 \
  --output outputs/profile/lidar_io.json

conda run -n kd_mm_beam kd-sensing-training-throughput --mode profile \
  --config configs/fusion/image_radar_gps_lidar_lightweight.yaml \
  --samples 32 \
  --output outputs/profile/fusion_all_io.json \
  --csv-output outputs/profile/fusion_all_io.csv
```

输出会记录 dataset `__getitem__`、DataLoader batch wait、CPU 到 GPU transfer、forward、backward/optimizer、step 总耗时、samples/s、CUDA peak memory，以及实际 DataLoader、non-blocking transfer、AMP 和 cache 参数。
新增字段包括：

- `dataloader_splits.train/test`：实际 batch size、num_workers、persistent_workers、prefetch_factor、pin_memory 和 drop_last。
- `getitem_component_seconds`：image、radar、gps、lidar、mmwave 和 auxiliary targets 的 `__getitem__` 均值、P50、P95。
- `wait_vs_gpu_step`：DataLoader wait、transfer、forward、backward/optimizer 的总耗时比例、均值比例和 P95 尖峰对比。
- `progress` 与 `cache_policy`：当前 batch progress 状态和 cache policy，便于复现实验运行条件。

## Baseline 参考

这次只读探索看到的训练 split 约有 3610 个样本；image/radar/GPS/LiDAR 每类约 28880 次帧引用，但唯一帧约 4366 个，重复因子约 `6.6x`。小样本计时中，LiDAR `.mat` 到 BEV 约 `0.03s/sample`，radar/GPS 相对更轻。包含 image 的实验应直接 profile RGB/ImageNet 帧读取、CPU 到 GPU transfer 和模型 step。

优化前建议先保存一轮 profile JSON；预热 LiDAR cache 后，再用同一命令复测，重点看 `dataset_getitem_seconds`、`dataloader_wait_seconds` 和 `samples_per_second`。

## Cache 预热顺序

先生成统一 split，再预热 LiDAR cache：

```bash
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/sequences_ra_gps_lidar.yaml
conda run -n kd_mm_beam kd-sensing-preprocess --config configs/preprocess/lidar_bev_cache.yaml
```

训练和评估入口默认使用 `data.cache.policy: auto`：包含 LiDAR 的任务自动读取/写入 LiDAR BEV cache，
不包含 LiDAR 的任务不会访问对应 cache。包含 image 的任务使用 RGB/ImageNet 输入，不会访问 image cache。只想复用已有
cache 而不补写缺失文件时，使用 `read_only`：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_lightweight.yaml \
  -o data.cache.policy=read_only
```

可用策略为 `off`、`read_only`、`auto`、`rebuild`，也可以按模态覆盖：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_lightweight.yaml \
  -o data.cache.policy=read_only \
  -o data.cache.lidar.policy=auto
```

四任务并行时不要让每个训练进程同时写同一批 LiDAR BEV cache。先用推荐器检查覆盖率：

```bash
conda run -n kd_mm_beam kd-sensing-training-throughput --mode recommend \
  --config configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml \
  --parallel-runs 4 \
  --cpu-count 32
```

如果输出的 `cache.lidar.coverage` 低于阈值，先运行 LiDAR cache 预热命令，并保留
`data.cache.policy=auto`；覆盖率足够后再使用推荐器输出的 `data.cache.policy=read_only`。

## Cache 复用与失效

可以长期复用：LiDAR BEV、radar RA/DA 预生成结果、beam label cache。训练参数一般不影响这些缓存，包括 `lr`、`epochs`、`batch_size`、`num_workers`、`seed`、模型结构、loss、scheduler 和输出目录。

需要新 cache 或清理旧 cache 的情况：

- 原始 jpg、LiDAR `.mat`/`.npy`、radar `.npy`、GPS txt 或 beam txt 内容变化。
- LiDAR BEV size、ROI、FoV、ground/background 过滤参数变化。
- radar FFT/RA/DA 生成参数变化。
- GPS feature mode 或坐标转换逻辑变化。

归一化统计更敏感：GPS scaler、LiDAR normalizer/stats、mmWave scaler、occlusion stats 和 position scaler
应与 train split、portion、feature mode、目标定义和预处理版本绑定。训练会把可复用 artifacts 写入
`artifacts/gps_scaler.npz`、`artifacts/lidar_normalizer.npz`、`artifacts/mmwave_scaler.npz`、
`artifacts/occlusion_target_stats.json` 和 `artifacts/position_target_scaler.npz`；四个 objective 使用同一
split 和预处理配置时，应优先复用这些 train-fitted artifacts。只有当 LiDAR BEV cache 覆盖率足够且不需要补写
缺失 cache 时，才把 cache policy 切到 `read_only`。

## DataLoader 与 AMP

并行跑多个实验时，先使用 `num_workers=2~4` 和 `prefetch_factor=1`。单个实验确认 CPU/I/O 不再阻塞后，再逐步调高：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_lightweight.yaml \
  -o data.dataloader.num_workers=8 \
  -o data.dataloader.prefetch_factor=2
```

默认启用 `training.transfer.non_blocking=true`。AMP 默认关闭，建议等 cache 预热并确认 DataLoader 不再饥饿后，再在 CUDA 上启用：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_lightweight.yaml \
  -o training.amp.enabled=true \
  -o training.amp.dtype=float16
```

新配置也支持按 split 覆盖 worker 生命周期：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml \
  -o data.dataloader.train_num_workers=3 \
  -o data.dataloader.test_num_workers=1 \
  -o data.dataloader.train_persistent_workers=true \
  -o data.dataloader.test_persistent_workers=false \
  -o data.dataloader.train_prefetch_factor=1 \
  -o data.dataloader.test_prefetch_factor=1 \
  -o output.progress.enabled=false
```

后台 tmux/tee 训练建议关闭 batch 级 tqdm：`output.progress.enabled=false`。这只影响 batch/epoch progress
输出，不影响 epoch metrics、`train_log.json`、checkpoint、TensorBoard scalar 或 `training_outputs.npz`。

## Train Epoch 子采样

快速调参或排障时，可以只缩短每个 train epoch 的 step 数，同时保留原 train CSV、dataset 初始化、normalizer/cache
语义和完整 validation/test split：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/fusion/image_radar_gps_lidar_mmwave_beam_supervised.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.fraction=0.1 \
  -o output.progress.enabled=false
```

也可以固定每个 epoch 的 train 样本数：

```bash
conda run -n kd_mm_beam kd-sensing-train --config configs/gps/lightweight.yaml \
  -o training.epoch_subsampling.enabled=true \
  -o training.epoch_subsampling.num_samples=256
```

`training.epoch_subsampling.fraction` 和 `num_samples` 二选一；`seed` 为空时使用 `experiment.seed`。
`rotate_each_epoch=true` 会按绝对 epoch 轮换无放回样本选择，因此 checkpoint resume 后同一 epoch 可复现。
`rotate_each_epoch=false` 用于固定小子集调试。运行 metadata 会记录完整 train 样本数、每 epoch 有效样本数、
seed、轮换设置以及是否退化为完整 epoch。

这项配置只减少 train epoch 的实际训练 step，不会缩小 validation/test split，也不会替代
`data.dataset.portion`。如果希望缩小 dataset 构建、split metadata、normalizer 拟合或 cache 预热输入，继续使用
`data.dataset.portion` 或准备更小的 split CSV。

## GPU 低利用率排查顺序

1. 先跑 `kd-sensing-training-throughput --mode profile`，看 `wait_vs_gpu_step.p95_spikes` 和 `phase_ratios.wait`。
2. 如果 wait 明显高于 forward/backward，检查 `getitem_component_seconds` 中最重的模态。
3. LiDAR 重时先看推荐器的 cache 覆盖率，必要时预热 cache，再复测。
4. 四实验并行时先限制 train/test worker 和 prefetch，并关闭 batch progress。
5. wait 降下来后再考虑 AMP 或 batch size；不要把模型结构当成第一怀疑对象。
