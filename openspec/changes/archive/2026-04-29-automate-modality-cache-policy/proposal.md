## Why

当前项目已经具备 image motion mask cache 和 LiDAR BEV cache，但训练配置仍要求用户手动理解并设置多个读写开关。并行运行大量单模态和 fusion 实验时，缺少统一的自动策略容易导致重复在线预处理、缓存未复用或多个训练进程同时写同一缓存。

## What Changes

- 新增统一的模态缓存策略配置，支持 `off`、`read_only`、`auto` 和 `rebuild` 模式。
- 训练和评估入口在构建 dataset 前根据启用模态自动解析 image/LiDAR cache 目录、启用读取已有缓存，并在策略允许时写入缺失缓存。
- 对不包含 image 或 LiDAR 的任务，不访问对应 cache 配置或原始文件，避免非相关模态阻塞训练。
- image motion mask 和 LiDAR BEV cache 写入必须使用原子写入或等价保护，降低并行实验写同一 cache 的竞争风险。
- 运行输出必须记录实际生效的 cache policy、cache 目录、是否启用读写和参数 hash 目录，方便复现实验。
- README 增加自动缓存策略说明和推荐命令，明确单模态与任意 fusion 组合的 cache 复用边界。

## Capabilities

### New Capabilities

- `automated-cache-policy`: 定义训练/评估入口如何按模态自动启用、读取、写入和记录预处理 cache。

### Modified Capabilities

- `modality-aware-data-loading`: 增加 dataset 在自动 cache policy 下按启用模态访问 cache 的行为要求。
- `lidar-preprocessing`: 增加 LiDAR BEV cache 在自动写入和并发写入场景下的安全要求。
- `experiment-workflow`: 增加训练/评估运行记录实际 cache policy 和 cache 目录的要求。

## Impact

- 主要影响 `src/kd_sensing/config/defaults.py`、`src/kd_sensing/engine/builders.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/evaluator.py`、`src/kd_sensing/data/datasets/scenario9.py` 和 `src/kd_sensing/data/transforms.py`。
- 需要补充配置解析和测试，覆盖不同 policy、不同模态组合、cache hit/miss 和并发写入保护。
- 不改变模型结构、loss、指标、checkpoint 格式或默认训练超参；自动策略只影响预处理 cache 的读写行为。
