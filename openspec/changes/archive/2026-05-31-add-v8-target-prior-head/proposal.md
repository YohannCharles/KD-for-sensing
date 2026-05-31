## Why

当前 `v7_shared_physical_private_residual` 在 MMW sensor-assisted LOSO quick validation 中暴露出明显的 source prior collapse：source logits 高度集中在 source 高频 beam 33/34/35，而 target crossroad 真值主要集中在 47/48/49/50/52/54/55。继续只调很小的 private residual 正则不足以验证冻结 source backbone 后是否仍有可迁移的 target 可分信息。

本变更提出最小可验证的 `v8_target_prior_head`：在冻结 source backbone 的前提下，用 target-specific head、target support label prior calibration 与 coarse-to-fine 诊断头直接验证预测分布能否从 source 高频区拉回 target 主峰。

## What Changes

- 新增 HiST-Beam 模型变体 `v8_target_prior_head`，保留 v7 作为 baseline。
- 为 v8 增加 target adapter、target head、可训练 target prior bias、可学习或固定的 `beta_prior`，默认最终预测不再由 source logits 主导。
- 支持从 target_adapt labeled support labels 初始化 Gaussian-smoothed target prior，禁止使用 target_test label、beam_power、path 或 radio 字段参与初始化。
- 为 target adaptation 增加 v8 soft beam label loss、prior smoothness loss 和可选 coarse-to-fine sector/offset 诊断 loss。
- 新增 `v8_target_head_only` freeze policy，默认冻结 source backbone 和 source/shared head，只训练 v8 target branch、prior 参数和可选诊断头。
- 支持 v8 最小实验模式：target linear probe、target prior head、source prior only、target prior coarse-to-fine，并预留 prototype classifier 诊断接口。
- 在 source-only target eval 和 adapted target eval 后输出 prediction histogram、true histogram、top beams、mean absolute beam error 和 neighborhood hit metrics。
- 预留 source train long-tail 去偏配置入口，但不作为第一阶段必需实现，不影响旧实验默认行为。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `hist-beam-cross-scene-adaptation`: 增加 `v8_target_prior_head` 模型变体、target prior 初始化、v8 freeze policy、v8 adaptation loss、实验模式和 prediction histogram 产物要求。
- `soft-beam-label-training`: 明确 v8 target adaptation 可基于 hard beam label 生成 beam topology soft label 并作为 supervised loss 使用，且不得把该 loss 命名为 KD。

## Impact

- 影响模型与配置：`src/kd_sensing/models/fusion/hist_beam.py`、HiST-Beam variant 注册逻辑、`configs/hist_beam/*.yaml`。
- 影响训练与适配：`src/kd_sensing/engine/hist_beam_losses.py`、`src/kd_sensing/engine/hist_beam_adaptation.py`、`src/kd_sensing/engine/hist_beam_loso_stages.py`。
- 影响评估产物：source-only target eval 与 adapted target eval 的 `metrics.json`、predictions artifact、`prediction_hist.json` 和 LOSO summary。
- 影响测试：新增或扩展 v8 model forward、prior 初始化、防泄漏、freeze policy、loss composition 和 histogram artifact 的单元/集成测试。
- 不引入新的外部依赖；所有项目相关 Python 验证命令继续通过 `conda run -n kd_mm_beam ...` 执行。
