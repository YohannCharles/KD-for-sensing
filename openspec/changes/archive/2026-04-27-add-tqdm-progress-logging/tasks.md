## 1. 配置与日志结构

- [x] 1.1 在默认配置中新增 `output.progress.enabled`，默认启用训练进度条。
- [x] 1.2 在训练流程中初始化 `epoch_logs` 结构，保持现有 `history` 指标数组字段不变。
- [x] 1.3 调整 `train_log.json` 写入内容，使其同时包含既有历史指标和 epoch 级进度摘要。

## 2. 训练进度条实现

- [x] 2.1 在 `src/kd_sensing/engine/trainer.py` 中使用 `tqdm.auto.tqdm` 包装 epoch 迭代。
- [x] 2.2 在训练 dataloader 迭代中使用 `tqdm` 展示 batch 进度，并根据 `output.progress.enabled` 控制是否禁用显示。
- [x] 2.3 使用 `set_postfix` 更新训练损失、任务损失、蒸馏损失、训练准确率和学习率等关键状态。
- [x] 2.4 确保进度条关闭、早停和异常退出路径不影响 TensorBoard writer 关闭与已有产物保存。

## 3. 兼容性与行为保持

- [x] 3.1 确认 `training_outputs.npz`、训练曲线、checkpoint 和 TensorBoard 标量字段保持现有语义。
- [x] 3.2 确认关闭 `output.progress.enabled` 时不会创建可视化进度条，但仍写入 epoch 级进度摘要。
- [x] 3.3 确认 dry-run 配置仍能覆盖训练轮数、合成数据集和输出 run name。

## 4. 验证

- [x] 4.1 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/image/logits_kd.yaml --dry-run --override output.progress.enabled=false` 运行训练 smoke test。
- [x] 4.2 检查 dry-run 输出目录中的 `train_log.json`，确认包含既有历史指标和 `epoch_logs`。
- [x] 4.3 使用 `conda run -n kd_mm_beam python -m pytest` 运行项目测试，确认进度条改动没有破坏现有行为。
