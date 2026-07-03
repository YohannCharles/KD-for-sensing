## ADDED Requirements

### Requirement: configurable missing pattern sampler
系统 MUST 支持训练时可配置 missing pattern sampler，用于控制每个 batch 内 sample-wise available mask 分布。sampler MUST 复用统一 missing pattern API，并支持 `default`、`uniform`、`weak_single_oversample`、`sensing_only_oversample`、`missing_gps_oversample`、`curriculum_easy_to_hard` 和 `curriculum_hard_to_easy`。

#### Scenario: oversample weak single patterns
- **WHEN** 配置 `missing_pattern_sampler=weak_single_oversample` 且 `pattern_sampling_weights.radar_only=2.0`
- **THEN** sampler MUST 提高 `radar_only` 的采样概率
- **AND** 输出 mask MUST 为 `[B, M]` sample-wise availability mask

#### Scenario: curriculum 按 epoch 选择 pattern 集合
- **WHEN** 当前 epoch 落在 `curriculum_schedule.epochs_11_20`
- **THEN** sampler MUST 只从该 epoch 范围声明的 pattern 名称中采样
- **AND** 未匹配 schedule 时 MUST 回退到 default 或抛出清晰配置错误

#### Scenario: pattern sampling log
- **WHEN** 训练启用非 default sampler
- **THEN** 每个 epoch 的 pattern/count/ratio MUST 写入 `outputs/scene31/analysis/pattern_sampling_logs/{run_name}_pattern_counts.csv`
