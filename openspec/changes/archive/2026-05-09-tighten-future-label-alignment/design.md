## Context

根 specs 已经包含 future-only 标签契约：`prepare_labels()` 只使用 `target_beam[:, :num_pred]`，诊断导出第一个 horizon 表示 `t+1`。但 CRAF/MARF specs 仍描述 `num_pred + 1` 输出，训练流程 spec 仍要求旧输出多一个历史 slot 时自动截取，并要求缺失 feature 时用安全 fallback 继续运行。

当前代码也围绕统一 helper 做了较宽适配：`adapt_model_output()` 接受多个别名和 tensor-only 输出，`select_prediction_slots()` 对任意更长 logits 取最后 `num_pred`，viewer prediction payload 仍保留可省略的中间变量。这个变更不重写训练架构，而是把 future-only 后已经不再需要的兼容承诺从契约和测试中移除，让错误尽早暴露。

## Goals / Non-Goals

**Goals:**

- 让 CRAF/MARF 模型、训练 loss、诊断导出和 tests 全部使用精确 `num_pred` future horizon。
- 删除或收紧旧 `num_pred + 1` fusion head 兼容需求，避免新实验无意继续输出历史 slot。
- 收紧模型输出适配：dict 输出必须有明确 logits；需要 KD feature 的路径必须有真实 feature，不能用 logits 伪装。
- 让 viewer prediction export 的 distribution、future labels 和 confidence curves 同长，且第 0 行固定为 `t+1`。

**Non-Goals:**

- 不改变 Scenario 9 CSV、dataset 字段或 `prepare_labels()` 的 future-only 实现。
- 不重构 Gradio viewer 对外部 manifest 的缺失字段容错；只收紧模型预测导出器写出的新 payload。
- 不移除单模态时序模型的 tail selection，因为 image/radar/GPS/LiDAR/mmWave 仍会输出历史+未来占位的时间序列。
- 不提供旧 CRAF/MARF `num_pred + 1` checkpoint 的自动结构迁移。

## Decisions

1. **CRAF/MARF head 必须直接输出 `num_pred`。**

   CRAF/MARF 是显式 horizon query 模型，head 维度由 `self.horizon` 控制。新契约下 `self.horizon = self.num_pred`，主 logits、unimodal logits、router 权重和 residual 诊断都共享这个长度。替代方案是继续输出 `num_pred + 1` 并在 loss 中截断；这会保留无标签 slot，也会让 viewer/metrics 语义继续依赖隐藏裁剪。

2. **`select_prediction_slots()` 保留 tail selection，但语义改为“长时序模型对齐”。**

   单模态 GRU/Transformer 类模型会输出整个输入时间轴，训练仍需要最后 `num_pred` 个 future slot。因此 helper 继续接受 `T >= num_pred` 并返回 `logits[:, -num_pred:, :]`。但 specs 和 tests 不再把 `T == num_pred + 1` 称为旧历史 slot 兼容；如果 fusion head 输出 `num_pred + 1`，这是模型契约问题，而不是新训练流程保证支持的场景。

3. **输出适配按真实字段失败。**

   dict 输出只支持明确 logits 字段；训练需要蒸馏 feature 时必须从模型输出中取得真实 `input_features` / `output_features`。缺失 feature 时应在 RKD 或相关 loss 路径报错，而不是用 logits fallback 继续产生难以解释的结果。legacy 三元组继续保留，因为现有单模态和旧 fusion 模型仍使用 `(pred, features, output_features)`。

4. **viewer prediction export 写出精确 future payload。**

   `_sample_prediction_payload()` 使用已由 `select_prediction_slots()` 对齐后的 logits/probs 和 `prepare_labels()` 输出。导出的 `confidence_curves[modality]`、`beam_distribution[modality].prob/logit` 和 `prediction.modalities[modality].future_labels` 长度必须一致；不再进行 `probs[1:]` 或 `labels[1:]` 风格兼容。

5. **测试从兼容行为转为契约行为。**

   CRAF/MARF tests 断言 head horizon 等于 `num_pred`。model output adapter tests 覆盖缺失 logits、缺失 feature 的失败语义。viewer prediction tests 覆盖 distribution 第 0 行和 manifest `label.future_beams[0]` 都表示 `t+1`。

## Risks / Trade-offs

- [Risk] 旧 `num_pred + 1` CRAF/MARF checkpoint 在新代码下不再有评估兼容承诺。Mitigation：proposal 标记 breaking；旧实验用旧代码评估，或显式写迁移脚本。
- [Risk] `select_prediction_slots()` 仍对任意长 logits 做 tail selection，读者可能误以为 fusion 旧 head 仍被支持。Mitigation：spec 和 tests 明确 fusion head 必须精确输出 `num_pred`，tail selection 只作为长时序输出对齐 helper。
- [Risk] 移除 feature fallback 后，部分临时模型或测试 stub 会失败。Mitigation：测试 stub 补齐真实 feature，或将不需要 KD feature 的路径显式配置为 no-KD/logits-KD。
- [Risk] viewer 外部旧 manifest 仍可能带旧 shape。Mitigation：本变更只约束模型预测导出器的新 payload，不改变 viewer 读取外部 manifest 的容错策略。

## Migration Plan

1. 更新 specs，删除 `num_pred + 1` fusion horizon 和旧 slot 兼容要求。
2. 收紧 CRAF/MARF horizon、unimodal/subset/counterfactual loss 的 shape 断言。
3. 收紧 `adapt_model_output()` feature fallback 和相关测试。
4. 简化 viewer prediction payload 生成，补充 `t+1` 对齐测试。
5. 使用 `conda run -n kd_mm_beam pytest tests/test_craf_fusion.py tests/test_marf_training.py tests/test_modality_visual_diagnostics.py tests/test_training_io_workflow.py` 做定向验证，再运行可接受的 broader pytest 分组。

## Open Questions

无。该变更采用严格 future-only 契约，不再为旧 fusion horizon 保留隐式兼容承诺。
