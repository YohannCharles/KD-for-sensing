## 1. 配置默认值

- [x] 1.1 在 `src/kd_sensing/config/defaults.py` 的默认 `training` 配置中新增或统一 `early_stopping_metric: val_adba` 和 `early_stopping_mode: max`。
- [x] 1.2 在 `src/kd_sensing/config/canonical.py` 的 canonical 配置生成路径中写入 DBA early stopping 默认值。
- [x] 1.3 批量更新 `configs/**/*.yaml` 中启用 early stopping 的默认训练配置，确保默认指标为 `val_adba` 且方向为 `max`。
- [x] 1.4 保留命令行覆盖能力，确认 `training.early_stopping_metric` 和 `training.early_stopping_mode` 可被配置覆盖机制解析。

## 2. 训练循环实现

- [x] 2.1 在 `src/kd_sensing/engine/trainer.py` 增加 early stopping metric 解析 helper，支持 `val_adba`/`dba`、`top1_val_acc`/`val_acc` 和 `val_loss` 等别名。
- [x] 2.2 将 early stopping improvement 判断从硬编码路径改为使用配置指标、配置方向和 `training.min_delta`。
- [x] 2.3 当默认 DBA/ADBA 指标缺失时抛出包含指标名和配置建议的清晰错误。
- [x] 2.4 保持 `best_top1.pth` 的显式 Top-1 checkpoint 逻辑可用，但默认 `best.pth` 的保存语义改为 configured early stopping metric。

## 3. checkpoint 与恢复

- [x] 3.1 在 `last.pth` metadata 中写入 `early_stopping_metric`、`early_stopping_mode`、`best_early_stopping_value`、`best_early_stopping_epoch` 和 `epochs_without_improvement`。
- [x] 3.2 恢复训练时优先读取通用 early stopping metadata，并兼容缺少新字段的历史 checkpoint。
- [x] 3.3 在 epoch 日志、最终配置或训练输出中保留实际 early stopping 指标和方向，便于复现实验。

## 4. 测试与文档

- [x] 4.1 更新配置测试，验证默认配置和 canonical 配置不再默认使用 `top1_val_acc`，而是使用 `val_adba`/`max`。
- [x] 4.2 更新训练 I/O workflow 测试，验证 checkpoint metadata 包含通用 early stopping 状态。
- [x] 4.3 增加 metric alias 和 direction 单元测试，覆盖 DBA max、Top-1 max、loss min 以及缺失 DBA 的错误信息。
- [x] 4.4 更新 README 中 early stopping 默认指标、`val_adba`/`dba/val_adba` 含义和显式覆盖示例。
- [x] 4.5 运行 `conda run -n kd_mm_beam pytest tests/test_student_configs.py tests/test_training_io_workflow.py`。
- [x] 4.6 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/image/no_kd.yaml --dry-run`，确认最终配置与日志记录 DBA early stopping 默认值。
