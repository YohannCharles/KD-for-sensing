## 1. TensorBoard writer 改造

- [x] 1.1 扩展 `_write_tensorboard_scalars` 的调用参数，使其接收当前 `objective` 和 `output.tensorboard` 配置，并保持现有测试可用的默认值。
- [x] 1.2 新增或抽取 beam scalar 写入 helper，只在 `objective in {"beam", "multitask"}` 时写入 `beam/accuracy_train`、`beam/accuracy_val`、`beam/val_atop3`、`beam/val_atop5` 和 `beam/val_adba`。
- [x] 1.3 确保 beam scalar 写入复用 `_finite_float_or_none`，跳过缺失、`None`、`NaN` 或非 finite 值。
- [x] 1.4 增加 `output.tensorboard.legacy_accuracy_tags` 配置读取逻辑；未显式启用时不写 `accuracy/train`、`accuracy/val`、`accuracy/val_atop3`、`accuracy/val_atop5` 和 `dba/val_adba`，启用时恢复历史写入。

## 2. Metric alias 与文档

- [x] 2.1 扩展 `_EARLY_STOPPING_METRIC_ALIASES`，支持 `beam/accuracy_val`、`beam/val_top1`、`beam/val_adba` 等 objective-specific beam 别名，并保留历史 `accuracy/*`、`dba/*` 别名。
- [x] 2.2 更新 README 中 TensorBoard 标量说明，把 `beam/*` 记录为推荐入口，并说明 `accuracy/*` 是 legacy 兼容 tag。
- [x] 2.3 在 README 中说明 `beam/*` 只包含 beam objective 和 multitask 的 beam 分任务，不包含 occlusion-only/position-only 的诊断性 beam accuracy。

## 3. 回归测试

- [x] 3.1 更新 `tests/test_training_io_workflow.py` 的 FakeWriter 测试，验证 beam objective 默认写入 `beam/*` 且不写 legacy `accuracy/*`。
- [x] 3.2 增加 occlusion-only 和 position-only TensorBoard writer 测试，验证即使 history 中存在 finite `val_acc`、`val_atop3`、`val_atop5`、`val_adba`，也不会写入 `beam/*`。
- [x] 3.3 增加 legacy 开关测试，验证 `output.tensorboard.legacy_accuracy_tags: true` 时恢复写入历史 `accuracy/*` 和 `dba/val_adba` tag。
- [x] 3.4 增加 early stopping alias 测试，覆盖 `beam/val_adba -> val_adba`、`beam/accuracy_val -> val_acc` 和历史别名继续可用。

## 4. 验证

- [x] 4.1 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`，验证 TensorBoard writer、训练输出和 alias 回归。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_student_configs.py -q`，确认 objective 默认值和配置解析未被破坏。
- [x] 4.3 运行短训练 smoke：`conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_beam_no_kd.yaml -o data.dataset.portion=0.01 -o training.epochs=1 -o output.tensorboard.enabled=true`，检查新 run 的 TensorBoard tag 包含 `beam/*`。
- [x] 4.4 运行 `openspec status --change add-beam-metric-namespace`，确认 proposal、design、specs 和 tasks 均为 apply-ready。
