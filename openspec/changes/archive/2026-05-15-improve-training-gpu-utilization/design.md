## Context

当前四个五模态 no-KD 任务同时运行在 GPU0-3。现场采样显示：

- `nvidia-smi dmon` 中各 GPU SM 利用率大多为 0%，偶尔短脉冲到 60%-90%。
- 每个训练主进程约占 90%-100% CPU，且每个任务额外保留 train/test DataLoader worker；四任务并行后 Python worker 数量膨胀到几十个。
- 训练日志显示 steady-state batch 约 1.2-1.4 秒，个别 batch 会到 2-5 秒；这类尖峰符合 DataLoader 等待和 CPU 文件预处理抖动。
- 五模态每个 batch 会读取或处理 image、radar、GPS、LiDAR BEV/mmWave 等多个小文件；即使 LiDAR BEV cache 已存在，RGB 图像解码、文本/NumPy 文件读取、normalizer/scaler transform 和进度条日志仍会占 CPU 和 I/O。

现有项目已经有 `scripts/profile_training_io.py`、LiDAR BEV cache、non-blocking transfer、AMP 配置和训练吞吐文档，但还缺少针对“多实验并行”的 worker 生命周期控制、模态级瓶颈诊断和可直接套用的推荐覆盖参数。

## Goals / Non-Goals

**Goals:**

- 让 profile 能解释 GPU 利用率低的根因：DataLoader wait、模态级 getitem、worker 数量、日志输出和 GPU step 分解。
- 降低四实验并行时的 CPU/DataLoader 过度并发，避免 test loader workers 在训练期间长期空转占资源。
- 提供一组可复用的并行训练配置建议，优先保证 GPU 持续有数据，而不是盲目增加 workers。
- 让后台 tmux 训练默认可选择低噪声输出，避免 batch 级 tqdm 通过 `tee` 写入大量控制字符。
- 复用或预热 LiDAR BEV cache、LiDAR normalizer、mmWave/GPS scaler 等 artifacts，减少每个任务重复初始化成本。

**Non-Goals:**

- 不改变模型结构或四个 objective 的训练语义。
- 不把所有模态预处理一次性改成新的数据格式；大规模离线打包可作为后续优化。
- 不默认启用 AMP 或更大 batch size；这些作为 profile 后的建议项，避免改变现有基线可比性。

## Decisions

1. **先增强观测，再改默认行为。**
   - 决策：扩展 `profile_training_io.py` 输出模态级耗时、worker 生命周期、loader wait p95/p99 和 progress 日志状态。
   - 理由：当前现象明显是输入管线瓶颈，但 image、LiDAR、radar、mmWave 中哪一项最重需要量化，避免只调 `num_workers`。
   - 替代方案：直接把默认 `num_workers` 提高到 8 或 16。该方案在四实验并行时会进一步放大 CPU worker 和 I/O 争用。

2. **训练期间延迟或限制 test DataLoader worker。**
   - 决策：增加 DataLoader 构建策略，让 test loader 在验证阶段再创建，或至少允许 test split 使用更低 `num_workers` 和禁用 `persistent_workers`。
   - 理由：当前每个任务会同时保留 train/test worker，四任务并行后 worker 数量翻倍；test worker 只在 epoch 末验证需要。
   - 替代方案：保持两个 loader 都 persistent。该方案简单，但浪费 CPU 内存和调度资源。

3. **提供并行训练推荐器，而不是硬编码单一默认值。**
   - 决策：新增 helper 根据 `parallel_runs`、CPU 数、启用模态、cache 策略给出覆盖参数，例如 `num_workers`、`prefetch_factor`、`test_num_workers`、progress、AMP。
   - 理由：单实验和四实验的最佳参数不同；五模态和弱模态子集也不同。
   - 替代方案：全局修改默认配置。会影响已有实验可比性，也可能让小机器过载。

4. **把后台日志降噪作为性能配置的一部分。**
   - 决策：支持 `output.progress.enabled=false` 或更细粒度的 progress 更新间隔，并在并行训练推荐中默认建议关闭 batch 级 tqdm。
   - 理由：当前 tmux 命令通过 `tee` 持续写 batch 级进度条，日志包含大量控制字符；这会增加 CPU/终端/磁盘开销，也降低日志可读性。
   - 替代方案：保留当前输出。适合交互式单实验，不适合后台四实验并行。

5. **cache 复用优先于在线重复预处理。**
   - 决策：增加文档和检查，确认 LiDAR BEV cache、LiDAR normalizer 和 scaler artifacts 可被四个 objective 共享；需要时提供预热命令。
   - 理由：五模态 batch 的 CPU 成本高，在线重复生成或写 cache 会产生明显抖动。
   - 替代方案：每个训练自己按 `auto` 写 cache。首次运行方便，但并行时会争用同一 cache 目录并放大 I/O。

## Risks / Trade-offs

- **Worker 数降低导致单实验变慢** → 推荐器按并行数量和 CPU 数生成建议；单实验仍允许高 worker。
- **延迟创建 test DataLoader 影响验证代码路径** → 保留现有 `build_dataloaders` 兼容入口，新增 lazy/limited test loader 路径并加训练 smoke test。
- **关闭 batch tqdm 降低实时可见性** → 保留 epoch 级日志、TensorBoard 和 training_outputs；调试时仍可打开 progress。
- **AMP 改变数值轨迹** → AMP 只作为建议和可选覆盖，不改变默认 FP32 基线。
- **cache read_only 在 cache 未预热时失败或变慢** → 推荐器和文档必须先检查 cache 覆盖率，缺失时提示预热命令。

## Migration Plan

1. 扩展 profile 输出字段，不改变已有字段。
2. 增加 DataLoader split-specific worker 配置，默认保持兼容；并行推荐命令显式输出覆盖参数。
3. 增加低噪声 progress 配置和 README 说明。
4. 增加 cache 预热/检查流程。
5. 用 `conda run -n kd_mm_beam pytest ...` 覆盖 profile、DataLoader kwargs、训练 smoke 和 config 解析。
