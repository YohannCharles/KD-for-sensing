## Why

训练流程当前会在结束后保存 `train_log.json`，但长时间训练期间缺少直观的终端进度反馈，也没有把每个 epoch 的进度摘要同步沉淀到日志中。为训练程序加入 `tqdm` 进度条并记录关键进度信息，可以提升实验可观察性，便于排查卡顿、估算剩余时间和复现实验过程。

## What Changes

- 在训练循环中使用 `tqdm` 展示 epoch 和 batch 级进度。
- 进度条需要展示当前 epoch、batch 进度、训练损失、任务损失、蒸馏损失、训练准确率和学习率等关键状态。
- 将每个 epoch 的训练/验证摘要保存到日志文件中，确保训练完成或提前停止后能查看进度历史。
- 保持已有 `train_log.json`、`training_outputs.npz`、TensorBoard 标量日志和训练曲线输出语义兼容。
- 支持在非交互输出或关闭进度条配置时禁用 `tqdm` 显示，但不影响日志保存。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `experiment-workflow`: 扩展训练流程的输出要求，要求训练期间提供 `tqdm` 进度条，并将 epoch 级进度摘要保存到运行日志。

## Impact

- 主要影响 `src/kd_sensing/engine/trainer.py` 的训练循环、日志写入和可选配置读取。
- 可能补充 `src/kd_sensing/config/defaults.py` 中的输出配置默认值。
- 可能新增或调整训练 dry-run/smoke test，验证进度条可禁用、日志字段存在且不破坏现有输出。
- 依赖层面复用项目已有 `tqdm` 依赖，无需新增第三方包。
