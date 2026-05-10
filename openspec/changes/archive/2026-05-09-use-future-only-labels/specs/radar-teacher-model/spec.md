## MODIFIED Requirements

### Requirement: Radar-only 输入准备
系统 MUST 提供 radar-only 输入准备路径，从 batch 中读取 `radar_ra` 和 `radar_da`，在 channel 维拼接为雷达模型输入，并按现有预测窗口规则补齐未来占位帧。

#### Scenario: 准备 radar-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: radar`
- **THEN** 系统 MUST 使用 `radar_ra` 和 `radar_da` 构造雷达输入
- **AND** 系统 MUST 不要求图像输入参与模型 forward

#### Scenario: 雷达预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** radar-only 输入 MUST 包含最近 8 个雷达时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam
