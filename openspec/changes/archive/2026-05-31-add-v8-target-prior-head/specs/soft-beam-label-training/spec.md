## ADDED Requirements

### Requirement: V8 target adaptation beam topology soft labels
系统 MUST 支持在 `v8_target_prior_head` target adaptation 中基于 hard beam label 生成 beam topology soft label，并将其作为 supervised beam smoothing loss 使用。该 loss MUST 与 KD distillation loss 分离命名和记录。

#### Scenario: 从 target support hard label 生成 soft label
- **WHEN** `hist_beam.variant=v8_target_prior_head` 且 `hist_beam.v8.use_soft_beam_label=true`
- **THEN** 系统 MUST 基于 labeled target_adapt support hard beam label 和 `hist_beam.v8.soft_label_sigma` 生成 beam soft distribution
- **AND** 每个 soft distribution 的概率和 MUST 在数值容差内等于 1
- **AND** 该生成过程 MUST NOT 读取 target-side beam_power、RSS profile、path fields、CSI 或 target_test label

#### Scenario: V8 soft label loss 使用非 KD 命名
- **WHEN** v8 target adaptation 使用 beam topology soft label 计算 supervised loss
- **THEN** diagnostics MUST 使用 `hist/v8/loss_final_soft_ce`、`loss/beam_soft_target`、`loss/beam_smoothing` 或等价非 KD 命名
- **AND** diagnostics MUST NOT 将该 loss 记录为 `loss/kd_soft_label`、`loss/distillation` 或 teacher-student KD loss

#### Scenario: soft label 关闭时回退 hard CE
- **WHEN** `hist_beam.v8.use_soft_beam_label=false`
- **THEN** v8 supervised final loss MUST 使用 hard-label CE 或明确记录 supervised final loss 不可用原因
- **AND** evaluation Top-K、NRP 和 prediction histogram MUST 继续使用 hard beam label
