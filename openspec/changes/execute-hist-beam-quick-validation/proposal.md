## Why

已归档的 `add-hist-beam-cross-scene-adaptation` 已经把 HiST-Beam、LOSO split、target adaptation 和汇总产物纳入主规格，但当前执行闭环仍不够明确：`kd-sensing-hist-beam-loso --execute` 必须从 plan-only 变成可真实训练、适应、评估、汇总并给出快速验证结论的入口。

本变更把范围限定为一个可收敛的后续工作包：先跑 target scene 34 的最小矩阵，再扩展到 33/32/31，用少量 variants、budgets 和 seed 判断 adapter/prototype 是否值得继续投入。

## What Changes

- 修正 `kd-sensing-hist-beam-loso --execute` 的语义：
  - 不再只返回 planned run。
  - 按矩阵依次执行 source training、source-only target_test evaluation、target adaptation、adapted target_test evaluation 和 LOSO summary 写出。
  - 对 source checkpoint 复用、overwrite/resume、失败 run 和产物路径做可追踪记录。
- 定义 HiST-Beam quick validation 最小矩阵：
  - `quick_smoke` 只作为资源探针，默认运行 target scene `34`、单 variant、单 budget、短 epoch 和小数据比例，确保 CLI 能尽快暴露数据/训练路径问题。
  - 完整方法验证使用显式 `quick_validation` 配置，target scene 先执行 `34`，随后扩展到 `33/32/31`。
  - `quick_validation` variants 至少覆盖 `v0_flat`、`v3_decoupled`、`v4_adapter`、`v5_adapter_proto`、`v6_full_finetune`。
  - `quick_validation` label budgets 先覆盖 `0` 和 `10`，seed 先固定为 `0`。
  - DeepSense6G `quick_validation` 不启用数据抽样，`data.dataset.portion` 使用 `1.0`，训练轮数使用 `training.epochs: 40`，避免 1 epoch smoke 结果被误用作方法结论。
  - CLI/config 支持 `max_runs` 一类硬上限，用于资源探针或人工分段执行矩阵。
- 明确执行产物：
  - 每个 run 必须保存 `metrics.json`、target_test predictions、fold/split/sampling metadata 和配置快照。
  - adaptation run 必须记录 trainable parameter count、trainable ratio、adaptation time 和 prototype coverage 或不可用原因。
  - 矩阵结束后必须输出 LOSO summary CSV/JSON，并保留单次运行路径。
  - 每个 run/stage 在开始时就写出 running metadata；长 source training/adaptation stage 应写出进度文件，方便用户判断是否卡住。
  - 用户中断或 stage 失败时，已开始和未开始的 run 必须能进入 partial summary，未开始 run 标记为 `missing` 或等价不可判定状态。
- 增加快速验证结论产物：
  - 汇总 adapter/prototype 相对 source-only 和 full fine-tuning 的 Top-K、coarse/fine accuracy、efficiency 指标。
  - 标记每个 target scene、budget 和 variant 的胜负关系与缺失 run。
  - 输出简短结论，说明 `v4_adapter`/`v5_adapter_proto` 是否优于 source-only 和 `v6_full_finetune`。
- 增加执行级测试：
  - 覆盖 `--execute` 不再只返回 planned。
  - 覆盖无数据或缺失必要 CSV/模态时 preflight 明确失败。
  - 覆盖有最小测试数据时 smoke 能跑出 summary 文件。

## Capabilities

### New Capabilities

无。本变更不新增独立能力，继续基于已归档 change 合入的 HiST-Beam 与 LOSO 主规格。

### Modified Capabilities

- `cross-scene-loso-workflow`: 收紧 LOSO orchestration 的 `--execute` 执行语义、preflight 失败语义、quick validation matrix 和 summary/结论输出要求。
- `hist-beam-cross-scene-adaptation`: 收紧 HiST-Beam evaluation/adaptation run 的执行产物、efficiency 指标和 adapter/prototype 对比结论要求。

## Impact

- 代码：
  - `src/kd_sensing/cli/hist_beam_loso.py` 需要把 `--execute` 从计划输出改为实际编排执行。
  - LOSO orchestration 需要串联 source training、source-only evaluation、adapter/full fine-tuning adaptation、adapted evaluation 和 summary writer。
  - preflight 需要在训练前检查 scene 数据根目录、CSV、启用模态资源和可写输出目录，并在无数据时给出明确错误。
  - summary/diagnostics writer 需要输出 CSV/JSON 和快速验证结论。
- 配置：
  - `configs/hist_beam/quick_smoke.yaml` 需要表达轻量资源探针，避免默认 CLI 一上来执行长矩阵。
  - `configs/hist_beam/quick_validation.yaml` 或等价配置需要表达 target scene、variant、budget 和 seed 的最小方法验证矩阵，并使用 DeepSense6G 全量 portion 与 40 epoch。
  - 所有项目相关 Python 命令、测试和验证继续使用 `conda run -n kd_mm_beam ...`。
- 测试：
  - 增加 CLI execution、preflight failure、smoke execution 和 summary artifact 测试。
  - 继续避免真实大规模训练进入单元测试；smoke 使用最小 fixture、极小 epoch 或可注入 runner。
