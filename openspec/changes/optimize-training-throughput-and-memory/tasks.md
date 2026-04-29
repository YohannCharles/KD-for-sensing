## 1. Profiling 与基线

- [x] 1.1 新增 `scripts/profile_training_io.py` 或等价 CLI，支持指定配置、split、样本数、warmup、device 和输出 JSON/CSV。
- [x] 1.2 profile 输出 dataset `__getitem__`、DataLoader batch wait、CPU→GPU transfer、forward、backward、optimizer step、GPU memory 和 samples/s。
- [x] 1.3 为 image、radar、gps、lidar、fusion 典型配置各提供小样本 profile 命令示例，命令必须使用 `conda run -n kd_mm_beam ...`。
- [x] 1.4 记录优化前 baseline 到文档或运行日志示例，用于后续对比 image motion mask cache 和 LiDAR cache 效果。

## 2. Image Motion Mask Cache

- [x] 2.1 在默认配置中新增 image cache 配置项，包括 `image_motion_cache_dir`、`image_motion_use_cache`、`image_motion_write_cache`、`image_motion_cache_version`、Gaussian sigma 和阈值策略字段。
- [x] 2.2 在 `src/kd_sensing/data/transforms.py` 中抽出单 pair motion mask 构造函数，并实现参数 hash cache dir、cache path 和 metadata helper。
- [x] 2.3 扩展 `load_motion_masks`，优先读取 pair-level cache，cache miss 时按现有算法生成并在启用写入时保存 `uint8` mask。
- [x] 2.4 新增 image motion mask 预处理入口，按 train/test CSV 收集唯一相邻帧 pair，跳过已存在文件，显示 tqdm 进度，并写出 `metadata.json`。
- [x] 2.5 将 Scenario 9 dataset 的 image 参数接入新 cache 配置，保持返回 shape、dtype 和旧配置默认行为兼容。
- [x] 2.6 增加测试覆盖 cache hit、cache miss 写入、参数变化进入新目录、metadata 写出、读取后输出与在线计算一致。

## 3. LiDAR BEV Cache 预热

- [x] 3.1 扩展 `src/kd_sensing/preprocessing/lidar.py`，支持多个 CSV 或 train/test 配置，跳过已存在 `.npy`，支持 `overwrite` 和 tqdm 进度。
- [x] 3.2 为 LiDAR cache metadata 写入 BEV size、ROI、FoV、ground/background 参数、CSV 列表、生成数量、跳过数量和 cache dir。
- [x] 3.3 补充或更新 `configs/preprocess/lidar_bev_cache.yaml`，确保 train/test CSV 都能预热，并与训练配置的 `lidar_cache_dir` 和参数 hash 一致。
- [x] 3.4 增加测试覆盖预热 train/test、跳过已存在文件、参数 hash 分流和训练 dataset 读取预热 cache。

## 4. Beam Label Cache

- [x] 4.1 在 `Scenario9Dataset` 中新增 beam path 到整数 label 的轻量 cache，优先在初始化阶段扫描当前 split 唯一路径或按需 lazy fill。
- [x] 4.2 将 `__getitem__` 中的 `np.loadtxt + argmax` 替换为缓存读取，并保留缺失/非法 beam 文件的清晰错误。
- [x] 4.3 增加测试覆盖重复 beam path 只解析一次、`input_beam`/`target_beam` 维度不变、`num_pred=1` 合约不回退。

## 5. DataLoader、Transfer 与 AMP

- [x] 5.1 在 batch 准备函数中为 tensor `.to(device)` 增加可配置 `non_blocking`，并确保 `pin_memory=True` 时能从训练/验证路径传入。
- [x] 5.2 在默认配置新增 `training.amp.enabled`、`training.amp.dtype`、`training.amp.grad_scaler` 和 `training.transfer.non_blocking`。
- [x] 5.3 修改训练循环，CUDA AMP 启用时使用 `torch.autocast` 和 `GradScaler`，禁用或 CPU 时保持现有 FP32 路径。
- [x] 5.4 评估/验证路径支持与训练一致的 autocast 开关，指标计算输出保持现有结构。
- [x] 5.5 调整 canonical YAML 中显式 `num_workers: 8` 的配置，给并行实验默认使用 2 到 4 worker 和 `prefetch_factor: 1` 的稳定配置，并保留命令行覆盖说明。
- [x] 5.6 增加测试覆盖 DataLoader 参数解析、non-blocking 参数透传、AMP 配置解析和 CPU/禁用 AMP 回退。

## 6. 文档与验证

- [x] 6.1 更新 README 或实验说明，写清缓存可复用条件、失效条件、清理方式和推荐预热顺序。
- [x] 6.2 添加 image motion mask cache、LiDAR BEV cache 和 profile 的实际运行命令，命令必须使用 `conda run -n kd_mm_beam ...`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest -q tests` 或更精确目标测试，确认新增和既有关键测试通过。
- [x] 6.4 运行 `conda run -n kd_mm_beam python scripts/profile_training_io.py --config <fusion-config> --samples <N>`，确认 profile 输出可读。
- [x] 6.5 运行短训练 smoke test：`conda run -n kd_mm_beam python scripts/train.py --config <fusion-config> -o data.dataset.portion=0.01 -o training.epochs=1 -o output.tensorboard.enabled=false`，确认 cache、non-blocking 和 AMP 配置不会破坏训练路径。
- [x] 6.6 运行 `openspec validate --all`；若当前 `openspec` CLI 仍卡住，记录阻塞进程和手动文件校验结果。
