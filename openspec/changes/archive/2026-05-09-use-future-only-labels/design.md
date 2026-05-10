## Context

Scenario 9 序列 CSV 已经把历史 beam 和未来 beam 分开：`beam1..beamN` 对应输入历史窗口，`future_beam1..future_beamH` 对应未来标签。dataset 返回的 `target_beam` 形状也是 `[num_pred]`，语义已经是未来 beam。

当前混入 `beam8` 的位置在训练输入准备层：`prepare_labels()` 会将 `input_beam[-1]` 拼到 `target_beam[:num_pred]` 前面，导致 label 长度变为 `num_pred + 1`。后续 `select_prediction_slots()`、训练 loss、验证指标、CRAF/MARF horizon、viewer prediction export 都围绕这个旧长度建立假设。

## Goals / Non-Goals

**Goals:**

- 将训练、验证、评估和诊断的 label 统一为未来标签 `[t+1, ..., t+num_pred]`。
- 保持 `num_pred` 的含义为“未来预测步数”；`num_pred=3` 时 label 长度为 3。
- 保持 dataset 字段名称和 CSV 序列生成格式兼容，不要求重新生成 CSV。
- 同步更新 CRAF/MARF 等多 horizon 模型，使新训练产物直接输出 `num_pred` 个 slot。
- 更新测试，覆盖 `num_pred=1` 和 `num_pred=3` 的 label/slot 对齐。

**Non-Goals:**

- 不改变 `input_beam`、`target_beam` 的 dataset 返回结构。
- 不改变 `future_beam1..future_beamN` 在 CSV 中的列名和生成规则。
- 不提供旧 checkpoint 的无损结构迁移；旧 head 维度不匹配时仍依赖现有 strict-load 配置和用户显式选择。
- 不调整 beam 分类类别数、downsample 规则或 Top-K/DBA 指标定义。

## Decisions

1. **在 `prepare_labels()` 中删除历史 beam 拼接。**

   新实现直接对 `batch["target_beam"][:, :num_pred]` 应用 downsample 后返回，shape 为 `[batch_size, num_pred]`。这样 dataset、训练、验证和可视化都使用同一个未来标签源。替代方案是在 dataset 层删除或重命名字段，但 dataset 当前字段已经是正确的未来标签，改 dataset 会扩大兼容风险。

2. **`select_prediction_slots()` 改为选择 `num_pred` 个 slot。**

   训练和评估继续通过统一 helper 截取模型输出。若旧模型输出仍有 `num_pred + 1` 个 slot，helper 将取最后 `num_pred` 个 slot，与未来标签对齐；若新模型输出正好为 `num_pred`，则直接使用。替代方案是新增兼容参数区分旧/新标签语义，但这会让同一配置下的指标含义不稳定。

3. **输入 padding 逻辑暂不扩大未来步数。**

   现有 `prepare_gps_inputs()`、`prepare_radar_inputs()`、`prepare_lidar_inputs()`、`prepare_mmwave_inputs()` 使用 `max(num_pred - 1, 0)` 个 zero padding，让单模态时序模型输出长度覆盖最后 `num_pred` 个预测 slot。`prepare_image_inputs()` 当前使用 `num_pred` 个 zero padding，是既有实现差异；本变更只要求输出选择和标签对齐为 `num_pred`，不在本次 proposal 中重构单模态输入长度策略。

4. **CRAF/MARF 内部 horizon 改为 `num_pred`。**

   CRAF/MARF 的 prediction head、unimodal head、router/residual head 都以 `self.horizon` 控制输出 slot 数。新训练应直接生成 `[B, num_pred, C]`，避免再产生无标签的 `beam8` slot。替代方案是保持模型输出 `num_pred + 1` 并只在 loss 中截断，但会浪费 head 容量，并延续诊断输出中的旧语义。

5. **viewer prediction export 不再丢弃第一个 slot。**

   旧导出逻辑把第一个 slot 当成 current beam，所以使用 `probs[1:]` 和 `labels[1:]`。新语义下第一个 slot 就是 `t+1`，导出必须保留全部 `num_pred` 个 future distribution。

## Risks / Trade-offs

- [Risk] 旧 checkpoint 的 CRAF/MARF head 形状可能与新模型 horizon 不匹配。Mitigation：在 proposal 和任务中标明 breaking change；测试 strict-load 仍按既有 checkpoint 规则报错，用户可选择旧代码或非 strict 策略做迁移实验。
- [Risk] 某些诊断或文档仍假设 `num_pred + 1`。Mitigation：全仓搜索 `num_pred + 1`、`probs[1:]`、`labels[1:]`、`beam8` 和相关测试断言，作为实现任务的显式检查项。
- [Risk] 输入 padding 长度历史差异可能让读者误以为所有模态都必须改 padding。Mitigation：本变更只定义 label/slot 对齐契约，除非测试证明某个模型输出长度不足，否则不扩大输入 padding 改动范围。
- [Risk] 历史指标与新指标不可直接比较。Mitigation：运行产物和变更说明中强调新 label 不包含 `beam8`，旧实验结果需按旧语义解读。

## Migration Plan

1. 合入代码后，重新运行训练/验证测试，确认所有 labels、logits、unimodal logits 的时间维一致为 `num_pred`。
2. 对需要可视化的 checkpoint 重新导出 viewer predictions，确保 `beam_distribution` 的第 0 行对应 `t+1`。
3. 新实验从头训练或使用兼容的非 fusion checkpoint；CRAF/MARF 旧 checkpoint 若 head 维度不匹配，不做自动转换。
4. 如需回滚，恢复 `prepare_labels()`、`select_prediction_slots()` 和 CRAF/MARF horizon 的旧 `num_pred + 1` 语义，并使用旧 checkpoint/指标。

## Open Questions

无。当前决策采用未来标签 `[t+1, t+2, t+3]`，不包含 `beam8`。
