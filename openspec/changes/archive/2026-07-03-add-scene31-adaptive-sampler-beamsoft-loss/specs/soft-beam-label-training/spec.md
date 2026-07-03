## ADDED Requirements

### Requirement: Beam-neighborhood supervised CE loss
系统 MUST 支持 opt-in `beam_neighborhood_ce` supervised beam loss。该 loss MUST 基于 hard beam label 构造 beam-neighborhood soft target，并与普通 hard CE 混合；validation/evaluation MUST 继续使用 hard-label Top-K、DBA 和 primary metrics。

#### Scenario: circular Gaussian soft target
- **WHEN** `loss.type: beam_neighborhood_ce` 且 logits class 数为 `K`
- **THEN** loss MUST 根据 logits shape 自动使用 `K` 作为 beam 数量
- **AND** 对真实 label `y` 构造 `target_j = exp(-d(j,y)^2/(2*sigma^2))/Z`
- **AND** `circular=true` 时 `d(j,y)` MUST 使用 `min(|j-y|, K-|j-y|)`
- **AND** 每行 soft target 概率和 MUST 在数值容差内等于 1

#### Scenario: hard and soft CE mix
- **WHEN** `mix_ce=0.5`
- **THEN** loss MUST 计算 `0.5 * hard_CE + 0.5 * soft_CE`
- **AND** `mix_ce` MUST 表示 soft target loss 权重，不得反向解释
- **AND** ignore index label MUST 不参与 hard 或 soft loss 平均

#### Scenario: sigma and circular config
- **WHEN** 用户配置 `sigma` 或 `circular`
- **THEN** loss MUST 使用配置值构造 soft target
- **AND** `sigma` MUST 支持至少 `1.0`、`1.5` 和 `2.0`

#### Scenario: dtype and device support
- **WHEN** logits 位于 CUDA、fp16 或 bf16 dtype
- **THEN** soft target 构造和 soft CE MUST 保持在兼容 device 上
- **AND** soft target MUST 不需要梯度
- **AND** finite logits 与合法 label 不得产生 NaN 或 inf loss

#### Scenario: startup diagnostics
- **WHEN** 训练构建 `beam_neighborhood_ce`
- **THEN** 系统 MUST 打印或记录 `num_beams`、`sigma`、`circular` 和 `mix_ce`
- **AND** 该诊断 MUST 不改变 loss 数值

### Requirement: Label smoothing CE baseline
系统 MUST 支持 opt-in `label_smoothing_ce` supervised baseline，用于与 beam-neighborhood loss 做对照。

#### Scenario: label smoothing config
- **WHEN** 配置声明 `loss.type: label_smoothing_ce` 和 `loss.smoothing: 0.05`
- **THEN** 训练 MUST 使用普通 hard-label cross entropy label smoothing
- **AND** 该配置 MUST 不构造 beam-neighborhood soft target
- **AND** validation/evaluation MUST 继续使用 hard-label metrics
