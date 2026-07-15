## ADDED Requirements

### Requirement: Same-model temporal superset consistency
训练 runtime MUST 支持显式 opt-in 的 same-model temporal superset consistency。Training extension MUST 对同一样本使用 `M- subseteq M+` 的 partial student 与 stop-gradient superset teacher，共享一次 teacher forward 给所有启用的 consistency loss；teacher MUST 临时使用 eval mode，结束后恢复 primary model 状态，evaluation MUST 不执行 teacher branch。

#### Scenario: Superset teacher 无外部模型
- **WHEN** temporal superset consistency 启用
- **THEN** runtime MUST 只构建一个 primary model
- **AND** teacher output MUST 来自该 model 的在线 no-grad forward
- **AND** 系统 MUST 不读取 teacher checkpoint 或构建 distiller

#### Scenario: Disabled path 零开销
- **WHEN** KD、beam ranking 和其它 superset loss 均关闭
- **THEN** temporal operator MUST 不保存 superset input payload
- **AND** training extension MUST 不执行第二次 model forward

### Requirement: Confidence-gated soft-logit consistency
系统 MUST 支持温度化 soft-logit KL，并以 stop-gradient teacher correctness 和归一化预测熵形成每样本权重。Teacher 预测错误时权重 MUST 为零；高熵样本权重 MUST 不高于低熵样本；feature L2 MUST 保持独立且 S1 profile 中为零。

#### Scenario: 错误 teacher 不施加强一致性
- **WHEN** superset teacher Top1 与真实标签不同
- **THEN** 该样本对 confidence-gated KL 的权重 MUST 为零
- **AND** diagnostics MUST 记录 gate mean、active ratio、raw KL 和 weighted KL

#### Scenario: Temperature scaling 合法
- **WHEN** `temperature=2` 且至少一个 teacher 样本通过 gate
- **THEN** KL MUST 使用 teacher probability 与 student log-probability
- **AND** loss MUST 乘以 `temperature^2` 并按有效 gate 权重归一

### Requirement: Circular beam-risk monotonic ranking
系统 MUST 支持基于 circular beam distance 的 superset-to-partial ranking。对 64 beam 或配置类别数，风险 MUST 为预测概率对 `min(|b-y|, C-|b-y|)` 的期望；当 superset teacher stop-gradient 时，ranking loss MUST 为 `relu(R(M-) - R(M+) - tolerance)`，不得使用会在激活区间增大 partial student 风险的反向 hinge。

#### Scenario: Circular wraparound 正确
- **WHEN** 真实 beam 为 0 且候选 beam 为 `C-1`
- **THEN** circular distance MUST 为 1
- **AND** 不得使用线性距离 `C-1`

#### Scenario: Monotonic diagnostics
- **WHEN** beam ranking 启用
- **THEN** metrics MUST 记录 ranking loss、teacher/student risk、`student-teacher` risk gap、partial excess violation rate 和只读 superset-worse rate
- **AND** ranking weight 为零时 MUST 不改变 total loss

#### Scenario: 激活 ranking 降低 student 风险
- **WHEN** partial student 风险超过 superset teacher 风险与 tolerance 且执行一步有效梯度更新
- **THEN** 更新后的 student circular risk MUST 下降
- **AND** teacher logits MUST 保持 stop-gradient

### Requirement: Superset 方法保持 extension 边界
temporal superset teacher、confidence gate、beam ranking 和 diagnostics MUST 实现在 U-Mask training extension 或窄 helper 中，不得扩写 trainer/validator 主循环或复制 evaluation loop。

#### Scenario: 共享训练生命周期不增加 suite 分支
- **WHEN** S1 T1/T2/J1 配置运行
- **THEN** trainer MUST 仍通过通用 `TrainingExtension` hooks 调用新增行为
- **AND** checkpoint、optimizer、validation 和 finalization schema MUST 保持兼容
