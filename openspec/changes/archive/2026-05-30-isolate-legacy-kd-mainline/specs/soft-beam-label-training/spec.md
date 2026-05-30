## ADDED Requirements

### Requirement: Beam soft target 不等同于 KD
beam-aware soft label、angular soft target 和 beam smoothing target MUST 被视为 beam-space prior 或 supervised label smoothing，而不是 teacher-student KD。无 teacher 的 soft target loss MUST 不被命名、记录或汇总为 distillation loss。

#### Scenario: no-KD soft target 日志命名
- **WHEN** no-KD supervised training 使用 `target_beam_distribution` 或等价 beam soft target
- **THEN** loss diagnostics MUST 使用 `loss/beam_soft_target`、`loss/beam_smoothing` 或等价非 KD 命名
- **AND** diagnostics MUST 不生成新的 `loss/kd_soft_label` 或 `loss/distillation` 字段表示该监督项

#### Scenario: soft target metadata 记录来源
- **WHEN** batch 或 run metadata 记录 beam soft target 的来源
- **THEN** metadata MUST 区分 `source_power_oracle`、`gaussian_from_hard_label`、`angular_smoothing` 或等价来源
- **AND** 只有 teacher prediction distribution 才 MAY 标记为 KD soft target

### Requirement: KD soft target 与 beam soft target 可共存但必须分离
若 legacy KD baseline 同时启用 teacher distillation 和 beam soft target supervised loss，系统 MUST 分离 supervised soft-target loss 与 distillation loss 的配置、日志和 summary 字段。

#### Scenario: legacy KD 同时使用 beam soft target
- **WHEN** legacy KD baseline 的 supervised task loss 使用 beam soft target，且 distillation loss 使用 teacher logits 或 features
- **THEN** supervised loss MUST 记录为 beam/task loss
- **AND** teacher-student loss MUST 单独记录为 distillation loss
- **AND** total loss composition MUST 能区分两者权重

#### Scenario: evaluation 不使用 soft target 或 KD target
- **WHEN** validation 或 evaluation batch 同时包含 hard label、beam soft target 和可选 teacher output
- **THEN** hard-label Top-K、DBA、NRP 和 beam power 指标 MUST 使用 hard `target_beam`
- **AND** evaluation summary MUST 不用 soft target 指标替代 hard-label 主指标

### Requirement: 历史 KD 命名迁移
项目 MUST 为历史上带有 KD 命名但实际表示 beam soft label 的字段、配置或日志提供迁移路径。新代码 MUST 使用 beam/soft-target/angle smoothing 命名；旧字段若继续读取，MUST 作为兼容输入处理。

#### Scenario: 旧 kd_soft_label 字段兼容读取
- **WHEN** 历史 artifact 或配置中存在等价的 `kd_soft_label` 命名但其来源不是 teacher distribution
- **THEN** 系统 MAY 兼容读取该字段
- **AND** 新写出的 artifact MUST 使用 beam soft target 命名
- **AND** migration warning 或 metadata MUST 标明旧命名已退役
