## ADDED Requirements

### Requirement: LiDAR baseline 必须报告退化基线对比
LiDAR-only baseline 训练和评估 MUST 报告模型指标与退化基线的对比。退化基线至少 MUST 包含 majority-class baseline；当序列 beam 历史可用时，还 MUST 包含 last-beam baseline。

#### Scenario: 报告 majority-class baseline
- **WHEN** 用户评估 LiDAR-only baseline
- **THEN** 评估报告 MUST 包含每个预测 horizon 的 majority-class Top-1 baseline
- **AND** 评估报告 MUST 包含 LiDAR 模型每个 horizon 的 Top-1/Top-3 指标
- **AND** 报告 MUST 标明 LiDAR 模型是否超过 majority-class baseline

#### Scenario: 报告 last-beam baseline
- **WHEN** batch 或 dataset metadata 中可获得历史 beam label
- **THEN** 评估报告 MUST 包含 last-beam Top-1 和 Top-3 baseline
- **AND** 报告 MUST 标明 LiDAR 模型相对 last-beam baseline 的差距

### Requirement: LiDAR canonical 模型配置使用 modular BEV encoder
LiDAR teacher/student/no-KD/KD canonical 配置 MUST 使用修复后的 LiDAR BEV profile 和 `modular_sequence` + `lidar_cnn` encoder，并保持现有 logits/loss 输出契约不变。

#### Scenario: 构建 LiDAR teacher baseline
- **WHEN** 用户加载默认 LiDAR teacher/no-KD 配置
- **THEN** 系统 MUST 构建 `modular_sequence` LiDAR 模型
- **AND** 模型输入 MUST 是经过 baseline profile 处理的 `[B, T, C, H, W]` LiDAR BEV 张量
- **AND** 模型输出 MUST 继续兼容现有 `[B, T, num_classes]` logits 选择和 loss 计算路径

#### Scenario: LiDAR 模型不改变 future-only 对齐
- **WHEN** LiDAR 模型输出序列长度大于 `num_pred`
- **THEN** 系统 MUST 继续只使用最后 `num_pred` 个输出时隙对齐 `[t+1, t+2, t+3]` 标签
- **AND** 系统 MUST 不把历史窗口最后一个 beam 重新纳入训练 label
