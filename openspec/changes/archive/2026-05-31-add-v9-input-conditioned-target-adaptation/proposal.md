## Why

v8 quick validation 已经验证 `source logits` 在 crossroad target 上会造成明显 source prior collapse，`target-specific head + target prior` 能把预测拉回 target beam 区域；但 A3/A5 同时暴露出新的 target prior collapse，预测仍集中到少数 target support 高频 beam，尚未形成输入条件化的 target 判别。

当前四个 v8 run 还出现 `eligible_run_count=0`，说明实验合法性判定或 target-side oracle 使用边界需要先审计并修正。v9 应先让 quick validation 结果可用于主结论，再在 v8 基础上加入最小的 input-conditioned local calibration 与 anti-collapse 诊断，而不是继续扩展 coarse-to-fine 结构。

## What Changes

- 修正 MMW sensor-assisted quick validation 的 run eligibility 判定，区分真正使用 target path/radio oracle 的 run 与仅因 validator 条件过严而被误标 ineligible 的 run。
- 为 v8/v9 evaluation 增加 collapse 来源诊断：`target_logits only`、`prior only`、`target_logits + prior`、support prior/pred/true histogram KL、`beta_prior` 初末值、per-true-beam confusion 和 unique predicted beam 覆盖。
- 新增 HiST-Beam `v9_input_conditioned_target_adaptation` 变体或等价 v8 mode，在冻结 source backbone 的前提下组合 target logits、受限 global target prior 和 sample-conditioned prototype logits。
- 为 global prior 增加强度约束：`beta_prior` 上限参数化、可选 fixed beta、prior dropout，并在 metrics 中记录实际生效的 prior 权重。
- 实现 target support prototype logits，支持 beam-level prototype 与 sector-level prototype（优先 `sector_size=2/3`），作为 feature-conditioned local calibration 项参与 final logits。
- 可选加入 widened-prior marginal KL 和 consistency/dropout 约束，用于 unlabeled target_adapt 或 labeled support batch 的 anti-collapse regularization；不得强制预测分布均匀化。
- 新增 v9 最小实验矩阵配置：A3 collapse 来源诊断、P1-P4 prototype ablation、可选 U1/U2 unlabeled distribution regularization。
- 保留 v8 A2/A3/A4/A5 结果作为诊断 baseline；不把 A5 coarse-to-fine 作为下一阶段主线。
- 不引入新的外部依赖，不改变 v0-v8 默认行为。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 增加 v9 input-conditioned target adaptation、prototype logits、global prior 限制、anti-collapse loss、collapse diagnostics 和 v9 实验模式/产物要求。
- `mmw-cross-scene-adaptation-protocol`: 明确 sensor-assisted quick validation 的 eligibility 判定必须基于实际使用的 target-side oracle 与 split 防泄漏证据，避免无 oracle 的 v8/v9 run 被错误排除，也避免使用 path/radio supervision 的 run 混入主结论。

## Impact

- 影响模型与配置：`src/kd_sensing/models/fusion/hist_beam.py`、HiST-Beam variant/mode 解析、`configs/hist_beam/*.yaml` 与 quick validation override。
- 影响 adaptation/loss：`src/kd_sensing/engine/hist_beam_losses.py`、`src/kd_sensing/engine/hist_beam_adaptation.py`、`src/kd_sensing/engine/hist_beam_loso_stages.py`。
- 影响 protocol 与 summary：MMW quick validation planner/runner、eligibility checker、LOSO summary、run metadata 和 `quick_validation_conclusion.json`。
- 影响评估产物：`metrics.json`、`prediction_hist.json`、新增 `collapse_diagnostics.json` 或等价字段、prototype probe metrics、confusion artifact。
- 影响测试：新增或扩展 eligibility、防泄漏、v9 forward、prototype logits、prior cap/dropout、anti-collapse loss、collapse diagnostics 和 quick validation summary 过滤测试。
- 所有项目相关 Python 验证命令继续通过 `conda run -n kd_mm_beam ...` 执行。
