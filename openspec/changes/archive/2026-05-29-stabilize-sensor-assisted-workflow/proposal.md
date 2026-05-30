## Why

当前 MMW sensor-assisted / HiST-Beam 快速验证已经能跑通局部测试，但项目表面积检查已暴露新增脚本未纳入生命周期 inventory，且 target-sensitive auxiliary supervision、metric horizon 聚合和 MMW split 预处理边界仍存在让结果不可比或主结论被污染的风险。现在需要把这些问题收敛成一组稳定化要求，先修正可复现的 guardrail 失败，再防止后续实验继续在含糊边界上扩展。

## What Changes

- 修复项目表面积和入口生命周期漂移：新增或保留的 MMW 数据准备脚本、shell orchestration 必须在架构 allowlist 与 inventory 中说明用途、生命周期和删除/收敛条件，重复 wrapper 不得成为推荐入口。
- 收紧 sensor-assisted target-sensitive supervision 语义：target radio/path/beam_power 等字段必须区分 source auxiliary、labeled target opt-in supervision、unlabeled target 禁用和 target_test 仅评价用途；使用不允许字段的 run 必须失败或标记为不可用于主结论。
- 对齐 HiST-Beam / sensor-assisted summary 与 quick conclusion：summary 必须汇总 sensitive-field flags、eligibility 和 exclusion reason，quick conclusion 不得把 ineligible run 当作主结论证据。
- 统一验证和 subset metric horizon 聚合：普通验证、force-mask subset 验证和 standalone evaluate 必须使用同一组 selected metric horizons，不得在 subset top1 中退回 first-valid-slot 口径。
- 将 MMW radar CSV / split materialization 的预处理职责从训练 executor 私有 preflight 中移到公开 MMW 数据准备或 split utility；训练 preflight 只能调用稳定公开入口或读取已准备 artifact。
- 为 `hist_beam_loso_execution` 的继续增长补拆分边界，优先把 preflight、stage orchestration、summary/conclusion 和 matrix metadata 分到窄模块，并保持公开 CLI/产物兼容。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `project-architecture`: 收紧脚本入口生命周期 inventory、表面积 guardrail 和 HiST-Beam LOSO executor 热点拆分边界。
- `dataset-runtime-contracts`: 明确 labeled target auxiliary supervision 的 opt-in 规则和 target sensitive field usage metadata。
- `mmw-sensor-assisted-beam-prediction`: 明确 sensor-assisted 主结论 eligibility、target sensitive supervision 处理和 summary 字段。
- `hist-beam-cross-scene-adaptation`: 明确 quick validation conclusion 对 ineligible / leaked / no-op run 的排除和报告语义。
- `experiment-workflow`: 明确 validation、force-mask subset validation 和 standalone evaluate 的 metric horizon 聚合必须一致。
- `mmw-town10-dataset-preparation`: 将 MMW split / radar CSV materialization 定义为公开数据准备能力，避免训练 preflight 依赖 dataset 私有 helper。

## Impact

- 主要影响 `tests/test_architecture_boundaries.py`、`docs/project_surface_inventory.md`、`src/kd_sensing/engine/hist_beam_adaptation.py`、`src/kd_sensing/engine/hist_beam_loso_execution.py`、`src/kd_sensing/engine/training_metrics.py` 和 MMW 数据准备/manifest/split 相关模块。
- 预期新增或调整 focused tests，覆盖架构边界、target sensitive guard、quick conclusion eligibility、metric horizon 一致性和 MMW preflight 公共入口。
- 不改变公开训练、评估、预处理和 manifest CLI 参数；内部模块可拆分，公开产物字段保持向后兼容并新增机器可读 eligibility/usage metadata。
- 不移动、删除、压缩或提交 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或本地生成 CSV 产物。
