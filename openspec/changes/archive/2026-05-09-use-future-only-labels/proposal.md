## Why

当前训练标签会把历史窗口最后一个 beam（例如 `seq_len=8` 时的 `beam8`）拼到未来标签前面，实际优化目标变成 `[beam8, t+1, t+2, t+3]`。这会把“重建当前时刻”和“预测未来时刻”混在一起，导致训练、评估、诊断中的 horizon 语义不够清晰。

本变更将标签和预测 slot 统一为纯未来语义：`num_pred=3` 时只表示 `[t+1, t+2, t+3]`。

## What Changes

- **BREAKING**: `prepare_labels()` 不再把 `input_beam[-1]` 拼入 label；返回 shape 从 `[batch_size, num_pred + 1]` 变为 `[batch_size, num_pred]`。
- `select_prediction_slots()`、训练 loss、验证指标、KD feature 对齐、CRAF/MARF unimodal 辅助 loss 和诊断预测导出全部按 `num_pred` 个未来 slot 对齐。
- CRAF/MARF fusion head 的内部 horizon 从 `num_pred + 1` 改为 `num_pred`，使模型输出 slot 与新标签长度一致。
- 保持 dataset 返回的 `input_beam` 和 `target_beam` 结构不变；`target_beam` 继续来自 CSV 的 `future_beam1..future_beamN`。
- 更新相关测试和文档/spec 断言，明确 `num_pred=3` 对应 `[t+1, t+2, t+3]`，不包含 `beam8`。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `modality-aware-data-loading`: `prepare_labels()` 的标签语义从“历史最后一帧 + 未来标签”改为“仅未来标签”。
- `experiment-workflow`: 训练、验证、评估、KD 和输出 slot 选择从 `num_pred + 1` 个时隙改为 `num_pred` 个未来时隙。
- `radar-teacher-model`: radar-only 预测窗口对齐规则改为使用最后 `num_pred` 个输出 slot 与未来标签对齐。
- `lidar-modality-model`: LiDAR-only 预测窗口对齐规则改为使用最后 `num_pred` 个输出 slot 与未来标签对齐。
- `mmwave-modality-model`: mmWave-only 预测窗口对齐规则改为使用最后 `num_pred` 个输出 slot 与未来标签对齐。

## Impact

- 影响训练/验证/评估路径：`src/kd_sensing/engine/batch.py`、`model_output.py`、`trainer.py`、`validator.py`、相关脚本和指标输入。
- 影响 fusion 模型输出 horizon：`src/kd_sensing/models/fusion/craf.py`、`src/kd_sensing/models/fusion/marf.py`。
- 影响可视化预测导出：`src/kd_sensing/diagnostics/viewer_predictions.py`，未来分布不再跳过第一个 slot。
- 影响现有 checkpoint 兼容性：旧 checkpoint 若输出 `num_pred + 1` 个 slot，评估时会只截取最后 `num_pred` 个 slot；新训练产物的 head 维度可能与旧 CRAF/MARF checkpoint 不完全兼容。
- 不新增运行时依赖，不改变 CSV 生成格式或 dataset 字段名称。
