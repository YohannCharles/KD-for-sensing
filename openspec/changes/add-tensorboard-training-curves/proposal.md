## Why

当前训练流程会在运行目录中保存 JSON、NPZ 和静态训练曲线图片，但无法用 TensorBoard 实时查看或横向比较不同实验的训练曲线。为训练流程添加 TensorBoard event 记录，可以更方便地观察 loss、accuracy、learning rate 等指标随 epoch 的变化，并保留现有文件产物以兼容当前工作流。

## What Changes

- 训练流程在每个运行目录下写入 TensorBoard event 日志，用于记录训练和验证曲线。
- 记录现有 `history` 中已有的核心标量：训练总损失、任务损失、蒸馏损失、训练准确率、验证损失、验证准确率和学习率。
- 增加可配置的 TensorBoard 开关和日志子目录，默认启用且写入当前 run 目录下的固定子目录。
- 保留现有 `train_log.json`、`training_outputs.npz` 和静态曲线图片输出，不引入破坏性变更。
- 更新项目依赖和文档，使用户可以直接通过 TensorBoard 查看训练曲线。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `experiment-workflow`: 训练运行输出需要包含 TensorBoard 标量日志，并支持通过配置控制日志写入行为。

## Impact

- 影响训练主流程：`src/kd_sensing/engine/trainer.py`。
- 影响默认配置和示例配置：需要增加 TensorBoard 日志配置项。
- 影响依赖声明：需要确保 `torch.utils.tensorboard` 的运行依赖可用。
- 影响用户文档：需要说明 TensorBoard 日志位置和启动命令。
