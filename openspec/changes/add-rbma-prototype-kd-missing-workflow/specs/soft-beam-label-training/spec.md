## ADDED Requirements

### Requirement: Prototype alignment beam-neighborhood targets are supervised topology targets
Beam prototype alignment 中使用的 beam-neighborhood soft target MUST 复用 beam soft label 的 topology 语义，并 MUST 作为 supervised alignment target 记录。该 target MUST NOT 被记录为 legacy KD soft target 或 retired distillation loss。

#### Scenario: prototype target 使用 beam topology
- **WHEN** prototype alignment 根据 hard `target_beam` 生成 soft target
- **THEN** 系统 MUST 使用配置声明的 `beam_label_sigma`、`beam_label_circular` 和 `num_beams`
- **AND** target 概率和 MUST 在数值容差内等于 1

#### Scenario: prototype loss 使用非 KD 命名
- **WHEN** prototype alignment loss 被加入训练总损失
- **THEN** diagnostics MUST 使用 `loss/prototype_alignment`、`loss/prototype_modality`、`loss/prototype_supcon` 或等价非 retired 命名
- **AND** diagnostics MUST NOT 将该 supervised topology loss 记录为 `loss/distillation` 或旧 `loss/kd_soft_label`

### Requirement: Full-to-partial teacher logits are distinct from beam soft labels
Full-to-partial teacher stabilization 中的 teacher logits/probabilities MUST 与 beam-neighborhood soft label 分开记录。teacher guidance loss MUST 标记为 opt-in stabilization，不能替代 hard-label evaluation 指标。

#### Scenario: teacher guidance 与 soft label 分离
- **WHEN** 同时启用 beam prototype alignment 和 full-to-partial teacher stabilization
- **THEN** prototype soft target diagnostics MUST 记录 beam topology 来源
- **AND** teacher KL diagnostics MUST 记录 online full teacher 来源
- **AND** 二者 MUST 使用不同 loss 名称和 sample count

#### Scenario: evaluation 仍使用 hard label
- **WHEN** validation 或 evaluation batch 包含 prototype soft targets 或 teacher outputs
- **THEN** hard-label Top-K、DBA 和 primary metric MUST 继续使用 hard `target_beam`
- **AND** summary MUST 不用 prototype target accuracy 替代 hard-label 指标
