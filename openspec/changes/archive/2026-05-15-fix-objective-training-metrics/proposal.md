## Why

当前 objective-aware 多任务实验已经能同时训练 beam、occlusion 和 position head，但默认 `multitask` 权重把 position 降到 `0.01`，导致“多任务”实际更接近 beam+occlusion 训练，不能公平评估三任务协同。训练日志还会把未启用任务的指标写成 `0.0`，容易把占位值误读为真实性能。

## What Changes

- 将 objective-aware canonical `multitask` 默认权重调整为 beam、occlusion、position 三个任务等权，确保三任务默认都能对共享 backbone 产生同等优化信号。
- 保持单任务 objective 的默认 early stopping 互不相同：beam 使用 `val_adba/max`，occlusion 使用 `val_occlusion_blocked_f1/max`，position 使用 `val_position_rmse/min`，multitask 使用明确的 multi-objective 主指标或加权总 loss。
- 修正训练历史、TensorBoard 和 `train_log.json` 的 inactive metric 语义：未启用或未计算的任务指标不得写成可误读的 `0.0` 曲线。
- 让 validation/evaluation 输出显式区分真实计算的 auxiliary metrics、未启用 metrics 和 multitask 加权 loss，便于解释 checkpoint 和 TensorBoard 曲线。
- 增加配置、loss、early stopping、日志和 TensorBoard regression tests，覆盖 objective-aware all/strong/weak 配置。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `configurable-multimodal-fusion`: objective-aware fusion canonical 配置的 `multitask` 默认 loss 权重改为三任务等权，并要求配置产物清楚记录解析后的权重。
- `experiment-workflow`: objective-specific early stopping、训练历史、TensorBoard 和验证指标输出必须区分 active metrics 与 inactive metrics，避免未启用任务的占位零值污染实验解释。

## Impact

- 影响 `src/kd_sensing/config/canonical.py`、`src/kd_sensing/engine/prediction_objectives.py`、`src/kd_sensing/engine/trainer.py`、`src/kd_sensing/engine/validator.py` 及 standalone evaluation 相关 objective metric 输出。
- 影响 objective-aware virtual config：`configs/fusion/<slug>_multitask_no_kd.yaml`、`strong_only_multitask_no_kd.yaml`、`weak_only_multitask_no_kd.yaml` 等。
- 影响 TensorBoard 曲线、`train_log.json`、`training_outputs.npz` 和 `metrics.json` 中 inactive objective metrics 的表示方式。
- 需要更新 README 或训练说明，解释旧 runs 中 `0.0` auxiliary 曲线只是 inactive 占位，不能作为真实性能。
