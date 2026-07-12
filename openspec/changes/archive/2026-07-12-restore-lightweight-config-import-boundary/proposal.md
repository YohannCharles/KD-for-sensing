## Why

当前冷进程执行 `import kd_sensing.config` 会经时序缺失配置规范化路径加载 `torch`，实测耗时约 1.55 秒、峰值内存约 616-620 MiB，违反现行 `project-architecture` 对配置轻量导入的 MUST 契约。该回归由近期时序缺失支持引入，应在更多配置工具和实验入口依赖这条路径前恢复并加上可持续护栏。

## What Changes

- 将时序缺失的纯配置契约与 tensor/mask runtime 解耦，使导入 `kd_sensing.config`、配置 normalization 和 configuration validation 不再因解析时序模式而导入 `torch`。
- 保持现有时序缺失模式、聚合语义、配置字段和 runtime helper 行为兼容；不新增配置格式、公开 CLI、第三方依赖或兼容 facade。
- 增加 fresh-process 轻量导入回归检查，验证配置包导入不会加载 `torch`、模型、dataset runtime、诊断渲染或训练主循环；检查不使用易受机器噪声影响的时间或内存硬阈值。
- 不改动时序 mask 采样、batch 变换、固定评估 mask cache、训练指标、checkpoint 或本地产物边界。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 明确时序缺失配置规范化属于轻量配置契约，不得通过 tensor runtime 模块把 `torch` 或训练运行依赖带入配置导入路径，并要求 fresh-process 回归场景覆盖该边界。

## Impact

- 主要影响 `src/kd_sensing/config/normalization.py`、`src/kd_sensing/data/temporal_missing.py`、`src/kd_sensing/data/difficulty/schema.py` 及承载纯时序配置契约的窄模块。
- 测试影响集中在 `tests/test_architecture_boundaries.py`、`tests/test_config_load_characterization.py` 和 `tests/test_temporal_window_missing.py`。
- 不改变 public console script、YAML 语义、模型/loss、数据 split、artifact schema、checkpoint schema 或默认输出路径。
