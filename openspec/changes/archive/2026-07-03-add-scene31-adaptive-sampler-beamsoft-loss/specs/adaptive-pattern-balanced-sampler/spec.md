## ADDED Requirements

### Requirement: Adaptive pattern-balanced sampler
系统 MUST 提供 opt-in adaptive pattern-balanced sampler，用于训练时按 missing pattern 困难度动态调整 sampling probability。默认未启用时 MUST 保持现有 sampler 行为不变。

#### Scenario: 配置启用 adaptive sampler
- **WHEN** 配置声明 `training.missing_pattern_sampler: adaptive_pattern`
- **THEN** 训练 MUST 使用现有 missing pattern mask helper 生成 `missing_mask`
- **AND** sampler MUST 支持 `alpha`、`temperature`、`ema_beta`、`score_mode`、`min_prob`、`max_prob`、`update_freq` 和 `warmup_epochs` 配置
- **AND** 未声明字段 MUST 使用 `alpha=0.5`、`temperature=1.0`、`ema_beta=0.9`、`score_mode=gap_to_full`、`min_prob=0.05`、`max_prob=0.40`、`update_freq=step` 和 `warmup_epochs=3`

#### Scenario: warmup 使用 uniform
- **WHEN** 当前 epoch 小于 `warmup_epochs`
- **THEN** sampler MUST 对所有核心 pattern 使用 uniform probability
- **AND** adaptive score 缺失不得导致训练失败

#### Scenario: gap_to_full score
- **WHEN** `score_mode=gap_to_full` 且 full pattern EMA loss 可用
- **THEN** 每个 pattern score MUST 等于 `max(0, EMA(loss_p) - EMA(loss_full))`
- **AND** full pattern score MUST 使用 full pattern EMA loss 作为 reference

#### Scenario: loss score
- **WHEN** `score_mode=loss`
- **THEN** 每个 pattern score MUST 等于该 pattern 的 EMA loss

#### Scenario: acc_gap warning
- **WHEN** `score_mode=acc_gap` 但训练循环没有可用 pattern-wise accuracy
- **THEN** sampler MUST 打印 clear warning
- **AND** sampler MUST fallback 到 uniform probability 而不是 crash

#### Scenario: probability clipping and normalization
- **WHEN** adaptive score 完整且 warmup 已结束
- **THEN** sampler MUST 计算 `(1-alpha) * q_uniform + alpha * softmax(score / temperature)`
- **AND** raw probability MUST clip 到 `[min_prob, max_prob]` 后重新 normalize
- **AND** 最终 probability 之和 MUST 在数值容差内等于 1

#### Scenario: epoch log
- **WHEN** adaptive sampler 完成一个训练 epoch
- **THEN** 训练 run dir MUST 写出 `adaptive_sampler_log.csv`
- **AND** 每行 MUST 包含 `epoch`、`pattern`、`ema_loss`、`score`、`sampling_prob` 和 `num_samples`
- **AND** 训练日志 MUST 打印包含 epoch 和 pattern probabilities 的简短摘要

#### Scenario: incomplete state fallback
- **WHEN** adaptive sampler state 缺少必要 EMA 或 pattern 集合为空
- **THEN** sampler MUST 使用 uniform probability
- **AND** sampler MUST 记录 warning
- **AND** 训练 MUST 继续运行
