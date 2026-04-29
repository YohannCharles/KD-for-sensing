# 训练吞吐与预处理缓存

当前优先优化的是训练输入管线，而不是模型显存。典型症状是 GPU 利用率偶尔升高、大多数时间接近 0%，DataLoader 和 CPU 预处理成为瓶颈。

## Profile 命令

所有命令都使用 `kd_mm_beam` 环境。建议先用小样本 profile，确认瓶颈在哪个阶段，再决定是否调 worker、cache 或 batch size。

```bash
conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/image/student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/image_io.json

conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/radar/student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/radar_io.json

conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/gps/student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/gps_io.json

conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/lidar/student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/lidar_io.json

conda run -n kd_mm_beam python scripts/profile_training_io.py \
  --config configs/fusion/image_radar_gps_lidar_student_no_kd.yaml \
  --samples 32 \
  --output outputs/profile/fusion_all_io.json \
  --csv-output outputs/profile/fusion_all_io.csv
```

输出会记录 dataset `__getitem__`、DataLoader batch wait、CPU 到 GPU transfer、forward、backward/optimizer、step 总耗时、samples/s、CUDA peak memory，以及实际 DataLoader、non-blocking transfer、AMP 和 cache 参数。

## Baseline 参考

这次只读探索看到的训练 split 约有 3610 个样本；image/radar/GPS/LiDAR 每类约 28880 次帧引用，但唯一帧约 4366 个，重复因子约 `6.6x`。小样本计时中，image motion mask 在线路径约 `0.11s/sample`，LiDAR `.mat` 到 BEV 约 `0.03s/sample`，radar/GPS 相对更轻。包含 image 的三模态训练慢于 `radar+gps+lidar`，与这个瓶颈一致。

优化前建议先保存一轮 profile JSON；预热 image/LiDAR cache 后，再用同一命令复测，重点看 `dataset_getitem_seconds`、`dataloader_wait_seconds` 和 `samples_per_second`。

## Cache 预热顺序

先生成统一 split，再预热 image 和 LiDAR cache：

```bash
conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/sequences_ra_gps_lidar.yaml
conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/image_motion_cache.yaml
conda run -n kd_mm_beam python scripts/preprocess.py --config configs/preprocess/lidar_bev_cache.yaml
```

训练时按需启用 cache：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_student_no_kd.yaml \
  -o data.dataset.image_motion_use_cache=true \
  -o data.dataset.lidar_use_cache=true
```

如果允许首次训练顺手补缺失 cache，可以额外打开写入：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_student_no_kd.yaml \
  -o data.dataset.image_motion_use_cache=true \
  -o data.dataset.image_motion_write_cache=true \
  -o data.dataset.lidar_use_cache=true \
  -o data.dataset.lidar_write_cache=true
```

## Cache 复用与失效

可以长期复用：LiDAR BEV、image motion mask、radar RA/DA 预生成结果、beam label cache。训练参数一般不影响这些缓存，包括 `lr`、`epochs`、`batch_size`、`num_workers`、`seed`、模型结构、KD 类型、loss、scheduler 和输出目录。

需要新 cache 或清理旧 cache 的情况：

- 原始 jpg、LiDAR `.mat`/`.npy`、radar `.npy`、GPS txt 或 beam txt 内容变化。
- LiDAR BEV size、ROI、FoV、ground/background 过滤参数变化。
- image size、Gaussian sigma、阈值策略、灰度化方式或 cache version 变化。
- radar FFT/RA/DA 生成参数变化。
- GPS feature mode 或坐标转换逻辑变化。

归一化统计更敏感：GPS scaler 和 LiDAR normalizer/stats 应与 train split、portion、feature mode 和预处理版本绑定。

## DataLoader 与 AMP

并行跑多个实验时，先使用 `num_workers=2~4` 和 `prefetch_factor=1`。单个实验确认 CPU/I/O 不再阻塞后，再逐步调高：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_rkd.yaml \
  -o data.dataloader.num_workers=8 \
  -o data.dataloader.prefetch_factor=2
```

默认启用 `training.transfer.non_blocking=true`。AMP 默认关闭，建议等 cache 预热并确认 DataLoader 不再饥饿后，再在 CUDA 上启用：

```bash
conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_rkd.yaml \
  -o training.amp.enabled=true \
  -o training.amp.dtype=float16
```
