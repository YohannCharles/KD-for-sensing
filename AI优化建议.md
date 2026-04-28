我按 explore 模式只做了调查，没有改代码。当前 OpenSpec 没有活动 change：`openspec list --json` 返回 `{"changes":[]}`。

**最主要的问题**

1. **Dataset 无条件加载 image + radar，导致 GPS/LiDAR-only 也在做大量无用 I/O**  
   [scenario9.py](/root/projects/KD-for-sensing/src/kd_sensing/data/datasets/scenario9.py:129) 里每个样本都会加载 motion mask、RA、DA、beam；之后才按需追加 GPS/LiDAR。  
   这意味着 `gps_teacher_no_kd`、`lidar_teacher_no_kd` 也在读相机和雷达数据。当前 CPU load 约 39，swap 已用 1.9/2.0 GiB，这个设计很可能是吞吐抖动的主因之一。

2. **当前并行训练资源调度不健康**  
   现在有 5 个训练主进程，每个主进程下大约 16 个 worker；`radar` 和 `gps` 还同时绑在 GPU 2。GPU 显存占用不高，但 GPU 利用率接近 0-20%，说明瓶颈主要在 CPU/I/O，而不是显存。  
   [builders.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/builders.py:40) 的 DataLoader 没有 `persistent_workers`、`pin_memory`、`prefetch_factor` 控制，worker 周期性重建会放大抖动。

3. **固定 run_name 会覆盖旧实验，当前 outputs 里已经出现“新旧文件混杂”**  
   [trainer.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/trainer.py:39) 用固定 `output.run_name` 创建目录且 `exist_ok=True`。训练中每个 epoch 会覆盖 `metrics.json`/checkpoint，但 `train_log.json` 只在训练结束才写入：[trainer.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/trainer.py:400)。  
   现在例如 `outputs/lidar_teacher_no_kd/final_config.yaml` 是 18:22 新写的，但 `train_log.json` 还是 16:11 的旧文件。分析结果时很容易误读。

4. **有两个后台等待任务可能永远等不到**  
   进程里有两个命令在等 `outputs/lidar_no_kd/checkpoints/best.pth`，但 canonical KD 配置实际指向 `outputs/lidar_teacher_no_kd/checkpoints/best.pth`，见 [configs/lidar/logits_kd.yaml](/root/projects/KD-for-sensing/configs/lidar/logits_kd.yaml:2)。  
   如果这些是你准备跑 LiDAR KD 的队列，它们现在基本是 stale waiter。

5. **不同模态使用的 train/test CSV 不一致，指标不能直接横向比较**  
   `train_seqs_RA/test_seqs_RA` 是 3201/1267 样本；GPS/LiDAR/GPS_LIDAR 是 3610/858 样本。当前 image/radar metrics 的 `total=1267`，GPS/LiDAR/all 的 `total=858`。  
   所以“单模态 vs 多模态”当前不是同一测试集对比。

**次级问题 / 风险**

- [scenario9.py](/root/projects/KD-for-sensing/src/kd_sensing/data/datasets/scenario9.py:157) 对 `target_beam` 做了 `.squeeze()`，如果 `num_pred=1`，后续 [batch.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/batch.py:29) 很可能维度错误。
- LiDAR 默认在线点云转 BEV，且当前配置没启用 cache。`load_lidar_bev_sequence` 支持 cache：[transforms.py](/root/projects/KD-for-sensing/src/kd_sensing/data/transforms.py:385)，但默认配置没有用起来。
- `outputs/evaluation` 是固定目录：[evaluator.py](/root/projects/KD-for-sensing/src/kd_sensing/engine/evaluator.py:16)，多次评估也会互相覆盖。
- 测试覆盖偏结构和 forward contract，目前 35 个测试，没有覆盖“只加载必要模态”“输出目录不混杂”“真实训练 I/O 性能”等问题。

**优先优化顺序**

1. 先把 Dataset 改成按 task/modalities 懒加载必要字段。这个收益最大。
2. 给训练输出加唯一 run id。
3. 统一评估 split。
4. 对 LiDAR 启用 BEV cache，避免每个 epoch 重复点云栅格化。
5. 清理后台 stale waiter，把 `lidar_no_kd` 等待路径改成 `lidar_teacher_no_kd`。