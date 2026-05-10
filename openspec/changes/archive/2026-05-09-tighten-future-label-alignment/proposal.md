## Why

`use-future-only-labels` 已经把训练标签收敛到 `[t+1, ..., t+num_pred]`，但根 specs 和部分实现契约仍保留 `num_pred + 1`、旧输出多一个 slot、缺失 feature 安全 fallback 等历史兼容描述。这些兼容层会让 horizon 语义再次变得含糊，也增加训练、诊断和测试路径的维护成本。

## What Changes

- **BREAKING**: CRAF 和 MARF fusion 模型的公开 horizon 契约收紧为直接输出 `num_pred` 个未来 prediction slot，不再要求或承诺输出 `num_pred + 1`。
- `select_prediction_slots()` 只保留“从长时序输出中取最后 `num_pred` 个 future slot”的单一语义，不再把 `num_pred + 1` 旧 head 作为兼容场景写入训练流程需求。
- 训练输出适配收紧为受支持的两种结构：legacy 三元组 `(logits, input_features, output_features)` 和 fusion dict 的明确核心字段；缺失蒸馏 feature 不再以 logits 伪装为 feature 继续运行。
- CRAF/MARF unimodal auxiliary loss、subset loss 和 counterfactual forward 的 logits/label 对齐必须使用精确的 `num_pred` 未来 horizon；多出的历史 slot 应暴露为模型契约错误，而不是静默裁剪。
- viewer 模型预测导出的 `confidence_curves`、`beam_distribution` 和 `prediction.future_labels` 必须与 future labels 同长，并且第 0 行固定表示 `t+1`；导出器不再承担旧 `probs[1:]`/`labels[1:]` 对齐兼容。
- 测试从“旧输出也能兼容”改为“新契约精确失败/精确对齐”，覆盖 CRAF、MARF、adapter、viewer prediction payload。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `experiment-workflow`: 收紧训练/验证/评估的输出适配与 slot 截取契约，移除旧 `num_pred + 1` 输出兼容需求和缺失 feature 安全 fallback 需求。
- `counterfactual-reliability-fusion`: 将 CRAF horizon 契约从 `N + 1` 改为 `N`，并要求辅助预测与主 future labels 精确对齐。
- `modality-adaptive-routing-fusion`: 将 MARF horizon 契约从 `N + 1` 改为 `N`，并要求 router、anchor/residual 和 logits 诊断使用同一 future horizon。
- `modality-visual-diagnostics`: 收紧模型预测导出 payload 的 horizon 语义，要求 distribution 第 0 行为 `t+1` 且与 `future_labels` 同长。

## Impact

- 影响 `src/kd_sensing/engine/model_output.py`、`trainer.py`、`validator.py`、`diagnostics/viewer_predictions.py`、`models/fusion/craf.py`、`models/fusion/marf.py` 以及相关脚本中对 `select_prediction_slots()` 的测试期望。
- 影响 CRAF/MARF 旧 checkpoint 的评估预期：旧 `num_pred + 1` fusion head 不再作为新契约下的受支持兼容路径；需要使用旧代码评估旧实验，或显式迁移 checkpoint。
- 不改变 dataset/CSV 字段、`prepare_labels()` 的 future-only 语义、单模态时序模型的输入 padding 策略、beam 类别数或 Top-K/DBA 指标定义。
