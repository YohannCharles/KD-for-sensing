## MODIFIED Requirements

### Requirement: 标签张量维度稳定
Scenario 9 dataset MUST 返回稳定维度的 `input_beam` 和 `target_beam`。单样本 `target_beam` MUST 保持形状 `[num_pred]`，batch 后 MUST 保持形状 `[batch_size, num_pred]`，包括 `num_pred=1` 的情况。`prepare_labels()` MUST 仅使用 `target_beam[:, :num_pred]` 生成训练标签，不得将 `input_beam` 的最后一个历史 beam 拼入 label。

#### Scenario: num_pred 为 1
- **WHEN** dataset 配置 `num_pred: 1` 且读取一个样本
- **THEN** 返回样本的 `target_beam` MUST 是一维张量且长度为 1
- **AND** DataLoader batch 的 `target_beam` MUST 是二维张量且第二维长度为 1
- **AND** `prepare_labels` MUST 返回形状 `[batch_size, 1]` 的未来标签
- **AND** `prepare_labels` MUST 不包含 `input_beam[-1]`

#### Scenario: num_pred 大于 1
- **WHEN** dataset 配置 `num_pred: 3` 且读取一个 batch
- **THEN** batch 的 `target_beam` MUST 保持 `[batch_size, 3]`
- **AND** `prepare_labels` MUST 返回 `[t+1, t+2, t+3]` 对应的 `[batch_size, 3]` 标签
- **AND** 训练、验证和评估指标 MUST 按 `num_pred` 个未来时隙计算
- **AND** 标签 MUST 不包含历史窗口最后一个 beam
