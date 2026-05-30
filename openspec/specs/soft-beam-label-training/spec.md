# soft-beam-label-training Specification

## Purpose
定义 beam soft label 的 batch 字段、source/target 域生成规则、soft-target supervised loss 消费方式，以及 hard-label 验证评估保持不变的训练契约。
## Requirements
### Requirement: Beam soft target batch contract

系统 SHALL 在启用 beam soft label 时为 beam selection batch 提供 `target_beam_distribution`，该字段 MUST 与 `target_beam` 的 future horizon 对齐，并表示每个 future slot 上所有 beam class 的概率分布。

#### Scenario: batch 包含 soft distribution
- **WHEN** Dataset 样本启用 soft beam label 并存在 `num_pred=3`、`num_classes=64`
- **THEN** 样本 MUST 包含 shape 为 `[3, 64]` 的 `target_beam_distribution`
- **AND** 每个 horizon 的分布和 MUST 在数值容差内等于 1
- **AND** 样本 MUST 继续包含 shape 为 `[3]` 的 hard `target_beam`

#### Scenario: hard label 指标和评估 loss 保留
- **WHEN** 训练或验证 batch 同时包含 `target_beam` 和 `target_beam_distribution`
- **THEN** 系统 MUST 使用 `target_beam` 计算 validation/evaluation loss、top-k、DBA、split 诊断和 hard-label 评价指标

### Requirement: Soft target generation

系统 SHALL 按训练域区分 beam soft target 来源：source 域 MAY 使用 beam power/RSS profile 归一化生成 soft target distribution；target 快速适应域 MUST NOT 读取或使用 target-side power/RSS profile，只能基于 hard beam label 和码本邻接关系生成 circular Gaussian soft distribution。

#### Scenario: source 域使用 beam power/RSS oracle
- **WHEN** source-domain future beam path 指向有效的 64 维有限 beam power/RSS 向量
- **THEN** 系统 MUST 将该向量转换为非负概率分布作为对应 horizon 的 soft target
- **AND** hard `target_beam` MUST 仍等于该向量的 argmax

#### Scenario: source 域使用 Gaussian fallback
- **WHEN** source-domain future beam power/RSS 向量缺失、维度错误、全零或非有限
- **THEN** 系统 MUST 基于 hard `target_beam` 和配置的 `sigma` 生成 Gaussian soft target
- **AND** circular 模式启用时，beam 0 与最后一个 beam MUST 按环形距离相邻

#### Scenario: target 域禁止使用 beam power/RSS oracle
- **WHEN** target-domain future beam path 指向有效的 beam power/RSS 向量
- **THEN** 系统 MUST NOT 读取或使用该 target-side power/RSS 向量生成训练 soft target
- **AND** 系统 MUST 基于 hard `target_beam`、配置的 `sigma` 和 circular beam distance 生成 Gaussian soft target
- **AND** beam 0 与最后一个 beam MUST 按环形距离相邻

### Requirement: Soft target supervised loss

系统 SHALL 在 beam soft target 可用且 soft target loss 启用时，使用 soft target distribution 计算主 beam supervised loss；若 soft target 不可用，MUST 回退到 hard-label loss。

#### Scenario: no-KD 主 loss 使用 soft target
- **WHEN** batch 包含 `target_beam_distribution` 且 `loss.soft_targets.enabled=true`
- **THEN** no-KD supervised loss MUST 消费 soft target distribution
- **AND** `loss/beam` 和 `loss/primary` MUST 记录 soft-target supervised loss

#### Scenario: KD 保持蒸馏逻辑
- **WHEN** 使用 logits KD 或 RKD 且 batch 包含 soft target
- **THEN** supervised task loss MUST 使用 soft target distribution
- **AND** distillation loss MUST 保持原有 teacher/student 逻辑

#### Scenario: validation 和 evaluation 不使用 soft target
- **WHEN** 验证 DataLoader batch 包含 `target_beam_distribution`
- **THEN** validation/evaluation loss MUST 使用 hard `target_beam`
- **AND** validation/evaluation top-k/DBA 指标 MUST 继续使用 hard `target_beam`

### Requirement: Configuration and fallback

系统 SHALL 提供配置开关控制 soft label 生成和 soft-target loss 消费，并允许显式关闭以复现 hard-label baseline。

#### Scenario: 显式关闭 soft target
- **WHEN** `loss.soft_targets.enabled=false` 或 `data.dataset.soft_beam_labels.enabled=false`
- **THEN** 系统 MUST 使用现有 hard-label supervised loss 路径

#### Scenario: 默认配置暴露参数
- **WHEN** 解析默认训练配置或 canonical beam objective 配置
- **THEN** 配置 MUST 包含 soft target 相关参数，包括 enable 开关、source、target_source、domain、sigma、circular、temperature 和 ignore index

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

