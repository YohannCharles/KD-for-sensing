## 1. 配置默认值与 objective metadata

- [x] 1.1 将 objective-aware canonical `multitask` 默认 `loss.objective.weights` 调整为 `beam=1.0`、`occlusion=1.0`、`position=1.0`。
- [x] 1.2 确认 `configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml`、`strong_only_multitask_no_kd.yaml` 和 `weak_only_multitask_no_kd.yaml` 的 virtual config 均解析为三任务等权。
- [x] 1.3 保留用户通过实体 YAML 或 CLI override 显式覆盖 multitask 权重的能力，并增加覆盖不影响其它权重的测试。
- [x] 1.4 在 final config、epoch log 或 runtime metadata 中记录 multitask 实际分任务权重，便于解释 `val_multitask_loss`。

## 2. Objective-specific early stopping

- [x] 2.1 修正默认 early stopping 语义：未设置 objective 或 `beam` 使用 `val_adba/max`，`occlusion` 使用 `val_occlusion_blocked_f1/max`，`position` 使用 `val_position_rmse/min`，`multitask` 使用 `val_multitask_loss/min`。
- [x] 2.2 更新配置解析测试，覆盖 objective 切换时 early stopping metric/mode 自动切换，以及显式覆盖 metric/mode 时仍尊重用户配置。
- [x] 2.3 强化 early stopping source validation：配置的 metric 如果当前 objective 未真实产出，训练必须报出清晰错误，而不是读取 inactive 默认零值。

## 3. Inactive metric 日志语义

- [x] 3.1 修改 validator/evaluator 输出，只将真实计算的 auxiliary metrics 写入 top-level metrics；inactive metrics 不得写成 `0.0`。
- [x] 3.2 修改 trainer history 和 epoch log：未启用任务的 optional metrics 使用 `None`、`NaN` 或省略表示不可用，不能记录为真实性能零值。
- [x] 3.3 修改 TensorBoard writer，只写入 active 且 finite 的 optional scalar；beam-only 不写 `position/rmse`，position-only 不写 `occlusion/blocked_f1`。
- [x] 3.4 保持 `training_outputs.npz` 对旧分析脚本的可读性；如保留固定 metric key，inactive slot 必须使用 `NaN` 或等价不可用表示。

## 4. 文档与实验解释

- [x] 4.1 更新 README 或训练说明，解释 objective-aware 单任务和 multitask 的默认权重、默认 early stopping 指标和指标方向。
- [x] 4.2 记录历史 runs 中 inactive auxiliary 曲线为 `0.0` 的解释方式，提醒不要把这些曲线当作真实性能。
- [x] 4.3 增加推荐 weight sweep 示例，说明用户可显式覆盖 `loss.objective.weights.*` 做非等权 multitask 消融。

## 5. 验证

- [x] 5.1 运行 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_student_configs.py -q`，验证 objective config、loss 权重和 early stopping 默认值。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q`，验证训练日志、TensorBoard 和 artifact 输出。
- [x] 5.3 运行 `conda run -n kd_mm_beam python scripts/train.py --config configs/fusion/image_radar_gps_lidar_mmwave_multitask_no_kd.yaml -o data.dataset.portion=0.01 -o training.epochs=1 -o output.tensorboard.enabled=true`，smoke 验证 multitask 等权、active metrics 和 TensorBoard tag。
- [x] 5.4 运行 `openspec status --change fix-objective-training-metrics`，确认 artifacts apply-ready。
