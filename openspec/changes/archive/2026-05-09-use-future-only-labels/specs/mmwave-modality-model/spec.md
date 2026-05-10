## MODIFIED Requirements

### Requirement: mmWave-only 输入准备
系统 MUST 提供 mmWave-only 输入准备路径，从 batch 中读取 `mmwave`，按现有预测窗口规则补齐未来占位时隙，并将结果传给 mmWave 模型。

#### Scenario: 准备 mmWave-only batch
- **WHEN** 训练、验证或评估流程处理 `experiment.task: mmwave`
- **THEN** 系统 MUST 使用 batch 中的 `mmwave` 构造 mmWave 输入
- **AND** 系统 MUST 不要求图像、雷达、GPS 或 LiDAR 输入参与模型 forward

#### Scenario: mmWave 预测窗口对齐
- **WHEN** `seq_length` 为 8 且 `num_pred` 为 3
- **THEN** mmWave-only 输入 MUST 包含最近 8 个 mmWave 历史时隙和 2 个未来 zero padding 时隙
- **AND** 验证和损失计算 MUST 使用最后 `num_pred` 个输出时隙与 `[t+1, t+2, t+3]` 标签对齐
- **AND** 输出时隙对齐 MUST 不包含历史窗口最后一个 beam
