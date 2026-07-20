# dynamic-router-physical-utility-alignment Specification

## Purpose
TBD - created by archiving change align-dynamic-router-physical-utility. Update Purpose after archive.
## Requirements
### Requirement: 互斥的融合决策目标
系统 MUST 为候选动态 Router 提供 `expected_utility`、`joint_hard_ce`、`power_soft_ce` 和 `power_top1_margin` 四种互斥的 fused-logit 决策目标，并在未声明时保持 `expected_utility` 历史默认行为。

#### Scenario: 选择单一目标
- **WHEN** 配置声明一个受支持的 fused decision objective
- **THEN** 系统仅计算该目标的加权训练 loss，并继续记录统一的连续 fused utility 诊断

#### Scenario: 拒绝未知目标
- **WHEN** 配置声明未知或非字符串的 fused decision objective
- **THEN** 系统在训练开始前 fail closed

### Requirement: Joint 硬标签决策监督
系统 SHALL 能在 same-availability Joint corrupted view 的 fused logits 上计算真实 beam label 交叉熵，且该目标 MUST NOT 要求 future beam power。

#### Scenario: 硬标签目标产生 Router 梯度
- **WHEN** Joint fused logits 的预测与真实 beam label 不一致
- **THEN** hard-label CE 对可训练 Router 参数产生有限梯度

### Requirement: 物理 beam-power 决策监督
系统 SHALL 在 `float32` 中验证和归一化非负有限 future beam power，并支持功率软标签交叉熵和最高功率 beam 对 hard negative 的 margin ranking；power target MUST detach 且不得进入模型 forward。

#### Scenario: 小量级线性功率不下溢
- **WHEN** future beam power 位于 MMW 线性功率量级且 fused logits 处于 AMP dtype
- **THEN** 功率归一化、软标签和 margin 计算保持有限非零结果与 Router 梯度

#### Scenario: Top-choice margin 只约束有效样本
- **WHEN** 样本的最高功率 beam 与 hard negative 存在正功率差
- **THEN** 系统计算按功率差缩放的 margin loss 并计入 active ratio

#### Scenario: 非法功率目标失败
- **WHEN** future beam power 缺失、维度不匹配、非有限、含负数或整行无正值
- **THEN** power 目标在训练开始或首个 batch fail closed

### Requirement: 决策目标可审计诊断
系统 MUST 分别记录所选 fused decision loss、连续 fused utility、目标 active ratio 和总 Router reliability loss。

#### Scenario: 日志区分目标和效用
- **WHEN** 任一候选完成一个训练 step
- **THEN** 日志同时包含 objective-specific decision loss 和与历史可比的 fused expected utility

