## 1. 配置和依赖

- [x] 1.1 在 `pyproject.toml` 中添加 `tensorboard` 依赖，确保 `torch.utils.tensorboard.SummaryWriter` 可用。
- [x] 1.2 在 `src/kd_sensing/config/defaults.py` 的 `output` 配置中新增 `tensorboard.enabled` 和 `tensorboard.log_dir` 默认值。
- [x] 1.3 确认现有 YAML 配置在不显式声明 TensorBoard 字段时仍能通过默认配置启用日志写入，并支持点号覆盖关闭或改名。

## 2. 训练写入逻辑

- [x] 2.1 在 `src/kd_sensing/engine/trainer.py` 中新增 TensorBoard writer 创建 helper，按 `output.tensorboard.enabled` 决定是否写入。
- [x] 2.2 在每个 epoch 完成验证并更新 `history` 后写入 `loss/train`、`loss/train_task`、`loss/train_distill`、`loss/val`、`accuracy/train`、`accuracy/val` 和 `learning_rate/main` 标量。
- [x] 2.3 确保 TensorBoard writer 在正常结束、early stopping 和异常路径下都会 flush 并 close。
- [x] 2.4 保持 `train_log.json`、`training_outputs.npz`、静态训练曲线和 checkpoint 输出行为不变。
- [x] 2.5 在训练验证结果中聚合 `ATop-3`、`ATop-5` 和 `ADBA`：`ATop-k` 使用所有有效目标时隙 Top-k accuracy 的平均值，`ADBA` 使用所有有效目标时隙 DBA 的平均值，且 DBA 沿用 Top-3 预测 beam。
- [x] 2.6 将 `ATop-3`、`ATop-5` 和 `ADBA` 写入 TensorBoard 标量，推荐 tag 为 `accuracy/val_atop3`、`accuracy/val_atop5` 和 `dba/val_adba`，并保持关闭 TensorBoard 时不执行相关写入。

## 3. 文档和验证

- [x] 3.1 更新 README 的训练输出说明，补充 TensorBoard event 日志位置和启动命令。
- [x] 3.2 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/image/no_kd.yaml --dry-run`，验证 dry-run 训练成功并生成 TensorBoard event 文件。
- [x] 3.3 使用 `conda run -n kd_mm_beam` 运行关闭 TensorBoard 的 dry-run 覆盖配置，验证训练仍成功且不会创建新的 TensorBoard event 文件。
- [x] 3.4 更新 README 的 TensorBoard 说明，补充 `ATop-3`、`ATop-5` 和 `ADBA` 的含义以及对应 tag。
- [x] 3.5 使用 `conda run -n kd_mm_beam python scripts/train.py --config configs/image/no_kd.yaml --dry-run` 运行 dry-run，并验证生成的 TensorBoard event 文件包含 `accuracy/val_atop3`、`accuracy/val_atop5` 和 `dba/val_adba`。
