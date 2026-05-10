## 1. 输出适配与 slot 契约收紧

- [x] 1.1 修改 `src/kd_sensing/engine/model_output.py`，让 `ModelOutput.input_features` 和 `ModelOutput.output_features` 能表达缺失状态，并移除用 logits 自动伪装 feature 的 fallback。
- [x] 1.2 收紧 `adapt_model_output()`：dict 输出必须提供受支持的 logits 字段；legacy 三元组继续支持，但三元组中的 feature 必须是真实 tensor 或在需要 feature 的训练路径中报错。
- [x] 1.3 保留 `select_prediction_slots()` 对长时序 logits 取最后 `num_pred` 个 slot 的行为，但更新 docstring、错误信息和测试，移除“旧 `num_pred + 1` fusion head 兼容”的表述。
- [x] 1.4 修改训练 KD/feature 对齐路径，使 RKD 或其它 feature-based loss 在 feature 缺失时清晰失败；no-KD 和 logits-KD 路径不要求 feature。

## 2. CRAF/MARF horizon 精确化

- [x] 2.1 确认 `src/kd_sensing/models/fusion/craf.py` 的主 prediction head 和 unimodal head 直接使用 `horizon = num_pred`，并移除相关注释或测试中的 `num_pred + 1` 期望。
- [x] 2.2 确认 `src/kd_sensing/models/fusion/marf.py` 的 logits、router、anchor/residual 和 unimodal 输出都使用 `horizon = num_pred`，并更新 MARF 定向测试。
- [x] 2.3 修改 `_unimodal_aux_loss()`，要求 unimodal logits horizon 精确等于 `num_pred`；遇到 `num_pred + 1` 或其它长度时直接报错，不再静默裁剪。
- [x] 2.4 检查 CRAF counterfactual forward、MARF subset training 和 validation subset 路径，确保它们只消费 `num_pred` future slots，且不会重建旧 current-slot 语义。

## 3. Viewer prediction payload 收紧

- [x] 3.1 简化 `src/kd_sensing/diagnostics/viewer_predictions.py` 的 `_sample_prediction_payload()`，直接使用已对齐的 probs/logits/labels 写出 payload。
- [x] 3.2 增加 shape 检查或测试断言，确保 `confidence_curves`、`beam_distribution.prob/logit` 和 `prediction.future_labels` 长度一致，且第 0 行表示 `t+1`。
- [x] 3.3 更新 `tests/test_modality_visual_diagnostics.py`，覆盖导出 payload 不执行 `probs[1:]`、`logits[1:]` 或 `labels[1:]` 旧偏移。

## 4. 文档、测试与验证

- [x] 4.1 更新 `tests/test_craf_fusion.py`、`tests/test_marf_training.py` 和相关模型测试，断言 CRAF/MARF head 输出长度精确等于 `num_pred`，并删除旧 `num_pred + 1` 兼容断言。
- [x] 4.2 更新 `tests/test_training_io_workflow.py` 或新增定向测试，覆盖 `prepare_labels()`、`select_prediction_slots()`、feature 缺失失败和 no-KD/logits-KD 可运行路径。
- [x] 4.3 全仓搜索并清理当前 specs、docs、tests 和注释中的新契约冲突文本：`num_pred + 1`、`N + 1`、`beam8`、`probs[1:]`、`labels[1:]`。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest tests/test_craf_fusion.py tests/test_marf_training.py tests/test_modality_visual_diagnostics.py tests/test_training_io_workflow.py` 验证定向改动。
- [x] 4.5 使用 `conda run -n kd_mm_beam pytest` 或项目可接受的分组测试完成回归验证，并记录任何因环境或耗时无法运行的测试。
