## Why

当前 shared runtime 的单模态输入准备路径存在 profile 路由错位风险：`radar` 分支可能读取 `input_profiles.gps`，`gps` 分支可能读取 `input_profiles.lidar`。这类问题很窄，但会影响 profile 驱动的数据契约、difficulty/reliability metadata 和后续 agent 自动修复的可信度。

## What Changes

- 修正 `prepare_task_inputs` 中单模态 `radar`、`gps`、`lidar`、`mmwave`、`csi` 的 input profile 读取规则，确保每个任务只读取同名 modality 的 profile。
- 为单模态 runtime profile routing 增加 focused tests，覆盖 radar/gps/lidar/mmwave/csi 的 profile 透传或默认行为。
- 在健康护栏中登记此类 shared runtime contract 的最小验证命令，防止未来新增模态或 profile 时再次错位。
- 不改变模型结构、dataset split、输出 schema、checkpoint schema 或训练 CLI。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `training-evaluation-runtime`: 增加单模态 runtime 输入准备必须按同名 modality profile 路由的契约。
- `project-health-guardrails`: 增加 shared runtime profile routing focused test 的健康护栏要求。

## Impact

- 主要影响 `src/kd_sensing/engine/runtime.py` 和 focused tests。
- 可能触碰 `tests/test_modality_difficulty.py` 或新增更窄 runtime contract test。
- 验证命令使用 `conda run -n kd_mm_beam pytest ...`，不读取真实 `dataset/`，不写入训练产物。
