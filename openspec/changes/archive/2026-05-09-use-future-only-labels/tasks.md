## 1. 标签与 slot 对齐

- [x] 1.1 修改 `src/kd_sensing/engine/batch.py` 的 `prepare_labels()`，只返回 downsample 后的 `target_beam[:, :num_pred]`，不再拼接 `input_beam[-1]`。
- [x] 1.2 修改 `src/kd_sensing/engine/model_output.py` 的 `select_prediction_slots()`，将所需 horizon 从 `num_pred + 1` 改为 `num_pred`，并保留旧输出多一个 slot 时取最后 `num_pred` 个 slot 的兼容行为。
- [x] 1.3 全仓清理训练、验证、评估和脚本中的手写 `num_pred + 1` slot 截取，改为统一 helper 或显式 `num_pred` 语义。

## 2. 模型与 loss 更新

- [x] 2.1 修改 CRAF fusion 模型 horizon，使 prediction head 和 unimodal head 输出 `[batch_size, num_pred, num_classes]`。
- [x] 2.2 修改 MARF fusion 模型 horizon，使 router、anchor/residual 和 unimodal 输出都按 `num_pred` 个未来 slot 工作。
- [x] 2.3 修改训练中的 KD feature 对齐、CRAF unimodal auxiliary loss、counterfactual loss 和 MARF subset loss 路径，确保 logits/labels/features 时间维一致为 `num_pred`。
- [x] 2.4 确认 radar、LiDAR、mmWave、GPS、image 和 fusion 任务的输入准备仍能提供足够输出 slot；除非测试证明不足，不扩大本次改动到 CSV 或 dataset 字段。

## 3. 诊断与文档

- [x] 3.1 修改 `src/kd_sensing/diagnostics/viewer_predictions.py`，导出的 `confidence_curves`、`beam_distribution` 和 `prediction.future_labels` 保留第一个 slot 作为 `t+1`。
- [x] 3.2 更新 viewer 或诊断相关测试/示例，明确 `label.future_beams[0]` 和 distribution 第 0 行都是 `t+1`。
- [x] 3.3 更新项目文档中关于 `num_pred + 1`、`beam8`、future horizon 或预测 slot 的描述，统一为 `[t+1, ..., t+num_pred]`。

## 4. 测试与验证

- [x] 4.1 更新 `tests/test_training_io_workflow.py`，覆盖 `num_pred=1` 时 `prepare_labels()` 输出 `[batch_size, 1]` 且不包含 `input_beam[-1]`。
- [x] 4.2 更新 `tests/test_craf_fusion.py`、`tests/test_marf_training.py` 和相关模型测试，断言 CRAF/MARF/select slot 输出长度为 `num_pred`。
- [x] 4.3 更新 modality visual diagnostics 测试，覆盖 prediction export 不再跳过第一个 future slot。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_craf_fusion.py tests/test_marf_training.py tests/test_modality_visual_diagnostics.py` 验证核心改动。
- [x] 4.5 使用 `conda run -n kd_mm_beam pytest` 或项目可接受的分组测试完成回归验证，并记录任何因环境或耗时无法运行的测试。
