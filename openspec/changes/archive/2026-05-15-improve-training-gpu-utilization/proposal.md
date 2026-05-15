## Why

当前五模态并行训练的 GPU 利用率呈现短脉冲，batch 约 1.2-1.4 秒且偶发 2-5 秒尖峰。现场进程显示每个训练主进程和 DataLoader worker 长时间占用 CPU，而 GPU 大量时间等待输入，说明瓶颈主要来自多模态 CPU 预处理、DataLoader 预取策略和后台日志输出。

这个问题现在需要处理，因为 beam、occlusion、position、multitask 四个五模态任务会同时运行在 GPU0-3；默认配置在单实验下保守可用，但在四实验并行时会把 image/radar/GPS/LiDAR/mmWave 文件读取、LiDAR BEV cache、test loader workers 和 tqdm 日志输出成倍放大。

## What Changes

- 增强训练吞吐 profiling，输出 DataLoader worker 数、train/test loader worker 生命周期、batch 间等待尖峰、模态级 `__getitem__` 耗时和日志输出开销。
- 增加并行训练推荐配置或 auto-tuning helper，根据并行实验数、CPU 数、启用模态和 cache 状态给出 `num_workers`、`prefetch_factor`、`persistent_workers`、`pin_memory`、progress 和 AMP 建议。
- 优化 DataLoader worker 策略，避免 test DataLoader 在训练期间长期保留 worker，减少四实验并行时的 CPU 进程膨胀。
- 提供五模态训练的 cache 预热与复用流程，优先复用 LiDAR BEV cache 和 train-fitted normalizer，避免每个任务重复做 streaming stats 和 cache write。
- 为后台 tmux 训练提供低噪声日志模式，默认关闭 batch 级 tqdm 刷新或降低刷新频率，保留 epoch 级日志和 TensorBoard。
- 不引入破坏性配置变更；现有配置继续可运行，新增行为通过配置或 helper 选择启用。

## Capabilities

### New Capabilities

### Modified Capabilities

- `training-throughput-optimization`: 扩展训练吞吐优化需求，从 profiling 覆盖到并行训练下的 DataLoader worker 控制、模态级瓶颈诊断、cache 复用建议、后台日志降噪和 AMP/worker 推荐配置。

## Impact

- 影响 `src/kd_sensing/engine/data_factory.py`、`scripts/profile_training_io.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py`、配置默认值和 README/docs。
- 可能新增 throughput recommendation 脚本或命令行入口，用于生成并行训练覆盖参数。
- 需要增加测试覆盖 DataLoader kwargs、profile 输出字段、progress 降噪、cache 复用配置和现有训练 smoke 测试。
