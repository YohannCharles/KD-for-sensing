## Why

当前训练产物已经包含 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json`、`train_log.json`、checkpoint sidecar 和 TensorBoard event，但这些信息分散在 run 目录、tmux 日志和运行进程中。随着 Multimodal-NF、Raymobtime、CSI hardening 等矩阵并行运行，用户很难快速判断每个 run 是正在训练、已完成、被 `Killed`、等待前置 checkpoint，还是只留下了启动产物。

本变更引入实验运行索引和状态汇总能力，把本地输出、日志和可选系统资源快照汇成一个只读、可复核的运行视图，降低后台训练和失败排查成本。

## What Changes

- 新增运行索引能力，扫描 `outputs/`、`logs/` 和可选进程/GPU 状态，生成结构化 run summary。
- 新增 CLI 入口或研究脚本入口，用于输出表格、JSON 和可选 CSV，支持按 dataset family、run state、objective、modality、时间范围过滤。
- 定义 run state 分类：`running`、`complete`、`started_no_metrics`、`partial`、`failed`、`killed`、`waiting`、`stale` 和 `unknown`。
- 定义日志解析边界：只读取本地 ignored 产物，不修改训练目录，不删除日志，不终止进程。
- 增强训练/评估运行产物契约，使运行开始、正常完成和异常退出时能尽量留下机器可读状态；异常退出无法捕获时，索引器必须能从已有 artifact 和日志推断状态。
- 新增资源快照字段，记录当前 GPU 显存、GPU 利用率、进程 RSS、系统内存和 swap 摘要，用于解释 `Killed`、低利用率或并行过载。
- 不引入外部服务、数据库或守护进程；默认实现保持本地文件扫描和命令行输出。

## Capabilities

### New Capabilities
- `experiment-run-index`: 定义本地实验 run 索引、状态分类、日志/产物扫描、资源快照和汇总输出契约。

### Modified Capabilities
- `experiment-workflow`: 增加运行状态 sidecar、异常/完成状态记录和索引器可消费字段的要求。

## Impact

- 可能新增 `src/kd_sensing/diagnostics/run_index.py`、`src/kd_sensing/cli/runs.py` 或 `tools/analysis/` 下的只读汇总入口。
- 可能新增 console script，例如 `kd-sensing-runs`，并更新 `pyproject.toml`、README 或 docs。
- 训练/评估 artifact writer 可能新增 `run_status.json` 或在 `train_log.json`/`metrics.json` 中补充 status 字段。
- 测试覆盖 run 状态推断、日志 `Killed` 识别、partial run 分类、CLI help 和 JSON 输出 schema。
