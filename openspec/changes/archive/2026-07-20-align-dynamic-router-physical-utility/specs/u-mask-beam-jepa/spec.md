## ADDED Requirements

### Requirement: Joint fused-logit 决策对齐
UMaskBeamJEPA 的候选动态 Router 配对训练 SHALL 将声明的互斥决策目标应用于 Joint corrupted view 的最终 fused logits，同时保持 control 与 Joint view availability 完全一致，且不得读取 corruption 类型、严重度或状态矩阵作为模型或 loss 特征。

#### Scenario: 配对视图应用决策目标
- **WHEN** dynamic Router paired Joint 训练启用且已构造相同 availability 的 control/joint 输出
- **THEN** loss 使用 Joint fused logits、beam label 及目标所需的可选 power 计算声明的决策监督

#### Scenario: Power 仅进入 loss
- **WHEN** 所选目标需要 future beam power
- **THEN** power tensor 只传入 loss，Router forward 输入与输出 schema 保持不变
