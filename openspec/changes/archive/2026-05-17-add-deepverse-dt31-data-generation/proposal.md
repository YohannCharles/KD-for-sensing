## Why

DeepVerse6G-DT31 需要先通过 generator 从场景参数生成可训练样本，而当前项目只有 DeepSense6G CSV 路径，无法直接得到 DT31 的通信、LoS、轨迹和传感器索引标签。先完成 Phase 1 数据生成可以把后续训练、baseline 和模态失衡实验建立在可复现的 cache 契约上。

## What Changes

- 新增 DeepVerse DT31 Phase 1 数据生成入口，读取 DeepVerse `ParameterManager`/`Dataset`，并稳健设置 DT31 场景、传感器和通信参数。
- 新增 beam codebook、future beam/blockage/trajectory 标签派生和 radar/weak wireless/noisy position cache 输出。
- 新增 manifest、labels、split、metadata、sanity report 输出，记录跳过样本原因和外部依赖状态。
- Phase 1.1 重新收紧 blockage 与 split 契约：blockage 必须经过 raw LoS/status 语义和类别可用性检查，默认 split 必须按 sequence/segment 或 contiguous temporal + embargo 防止滑窗泄漏。
- 新增配置文件和脚本命令，只覆盖数据生成，不实现训练 Dataset、模型或模态失衡实验矩阵。

## Capabilities

### New Capabilities
- `deepverse-dt31-data-generation`: 定义 DeepVerse6G-DT31 Phase 1 cache 生成契约，包括输入参数、输出 artifacts、标签语义、split 与 sanity checks。

### Modified Capabilities
- 无。

## Impact

- 新增 `src/kd_sensing/data/deepverse/` 下的 DT31 generator、label builder、codebook、split 和 sanity check 模块。
- 新增 `scripts/deepverse/generate_dt31_cache.py` 和 `configs/deepverse/dt31_generation.yaml`。
- 真实运行依赖外部 `deepverse` Python 包和 DeepVerse DT31 场景目录；项目依赖文件暂不强制加入 `deepverse`，脚本在缺包或缺场景时给出明确错误。
