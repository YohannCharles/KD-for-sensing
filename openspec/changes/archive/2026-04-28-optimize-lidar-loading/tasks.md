## 1. 配置与兼容解析

- [x] 1.1 梳理所有 LiDAR-only 和包含 LiDAR 的 fusion 配置，确认当前 `lidar_normalize`、`lidar_cache_dir`、`lidar_use_cache` 默认值和命令行覆盖行为。
- [x] 1.2 新增或调整 LiDAR 归一化配置解析，支持结构化 `lidar_normalization.enabled/mode/stats_path`，并保留 `lidar_normalize: true|false` legacy bool 兼容。
- [x] 1.3 将 LiDAR-only canonical 配置默认改为不触发训练集全局 z-score，同时保留显式启用 streaming stats 的覆盖能力。
- [x] 1.4 将包含 LiDAR 的 fusion canonical 配置同步改为新的懒加载归一化默认语义。

## 2. LiDAR Dataset 懒加载

- [x] 2.1 移除 `Scenario9Dataset.__init__` 中会遍历全训练 split 的 LiDAR normalizer 准备逻辑，确保初始化只解析 CSV、路径、列和小型配置。
- [x] 2.2 保持 `__getitem__` 按样本读取 LiDAR 点云或 BEV cache，并返回 `[seq_len, channels, height, width]` 的 `torch.float32` 张量。
- [x] 2.3 调整或限制 dataset 内部 `_lidar_bev_cache`，避免默认随样本访问无限增长为全训练集内存缓存。
- [x] 2.4 确保验证和评估 split 不会尝试在自身数据上 fit LiDAR normalizer。

## 3. 流式统计与缓存复用

- [x] 3.1 实现内存有界的 LiDAR streaming stats helper，逐样本或逐 batch 按通道累计 mean/std 所需的 sum、sumsq 和 count。
- [x] 3.2 支持将 LiDAR stats 保存为小型 `.npz` 或 `.pt` 文件，并能从配置指定路径加载复用。
- [x] 3.3 在 dataloader builder 或预处理路径中接入显式启用的 streaming stats，并在 `output.progress.enabled` 为 true 时显示独立进度或日志。
- [x] 3.4 保持 `.npy` BEV cache 按帧或按样本命中，不在 dataset 初始化阶段全量读取 cache 目录。

## 4. 测试

- [x] 4.1 增加 dataset 初始化测试，使用 monkeypatch 或轻量 fixture 验证启用 LiDAR 时初始化不调用全量 `_lidar_bev_for_index`。
- [x] 4.2 增加 LiDAR streaming stats 单元测试，验证统计结果与小样本直接计算一致，且不依赖 `np.concatenate` 全量拼接。
- [x] 4.3 增加 stats 文件复用测试，验证测试 split 使用训练 stats 且不会重新 fit。
- [x] 4.4 增加 LiDAR fusion 配置测试，验证启用 LiDAR 的 fusion 配置继承新的懒加载归一化默认语义。

## 5. 验证与文档

- [x] 5.1 使用 `conda run -n kd_mm_beam pytest` 运行相关测试，至少覆盖 LiDAR preprocessing、dataset、fusion config 和现有 student config 测试。
- [x] 5.2 使用 `conda run --no-capture-output -n kd_mm_beam python -u scripts/train.py --config configs/lidar/teacher_no_kd.yaml -o data.dataset.portion=0.01 -o training.epochs=1 -o data.dataloader.num_workers=0 -o output.tensorboard.enabled=false` 做 LiDAR-only smoke test。
- [x] 5.3 使用 `conda run --no-capture-output -n kd_mm_beam python -u scripts/train.py --config <包含 LiDAR 的 fusion 配置> -o data.dataset.portion=0.01 -o training.epochs=1 -o data.dataloader.num_workers=0 -o output.tensorboard.enabled=false` 做 fusion smoke test。
- [x] 5.4 更新 README 和 `LiDAR模态读取方案.md`，说明默认懒加载行为、`conda run --no-capture-output` 建议、BEV cache 使用方式和 streaming stats 复用命令。
