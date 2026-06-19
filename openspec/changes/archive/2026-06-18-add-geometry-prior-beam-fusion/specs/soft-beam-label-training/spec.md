## ADDED Requirements

### Requirement: DBA-aware supervised beam loss
系统 MUST 支持 opt-in DBA-aware supervised beam loss。该 loss MUST 利用 beam topology、calibrated class order 或 beam distance 奖励接近 target 的预测，但 validation/evaluation MUST 继续使用 hard `target_beam` 的 Top-K 和 DBA。

#### Scenario: distance-aware loss 使用 hard label 构造
- **WHEN** 配置启用 DBA-aware loss 且 batch 只有 hard `target_beam`
- **THEN** 系统 MUST 基于 hard target 和 beam topology 构造 distance-aware target 或 weighting
- **AND** loss diagnostics MUST 记录 topology mode、sigma/temperature、distance mode 和 class order

#### Scenario: evaluation 仍使用 hard target
- **WHEN** validation 或 evaluation batch 包含 DBA-aware soft target 或 beam topology target
- **THEN** validation/evaluation Top-K、DBA 和 primary metric MUST 使用 hard `target_beam`
- **AND** summary MUST 不用 soft target accuracy 替代 hard-label 指标

### Requirement: Geometry-prior loss logging is non-retired
Geometry-prior loss、DBA-aware loss 和 teacher-guided stabilization MUST 使用当前 supervised/objective 命名，不得复用 retired KD loss 命名或旧 distillation runtime。

#### Scenario: supervised beam smoothing 命名
- **WHEN** 训练使用 circular Gaussian、distance-aware CE、ordinal/EMD-style loss 或 focal circular loss
- **THEN** logs MUST 使用 `loss/beam_dba_aware`、`loss/beam_circular_soft_ce`、`loss/beam_distance_ce` 或等价非 retired 命名
- **AND** logs MUST 不将该 loss 记录为 `loss/distillation` 或旧 `loss/kd_soft_label`

#### Scenario: teacher guidance 命名
- **WHEN** 训练使用 teacher logits/probabilities 作为 stabilization
- **THEN** logs MUST 使用 `loss/teacher_guidance` 或 `loss/geometry_teacher_kl`
- **AND** run metadata MUST 标记 teacher guidance 为 opt-in stabilization，而不是 retired KD workflow

### Requirement: DBA-aware loss ablation isolation
DBA-aware loss MUST 可独立开关，并 MUST 支持与 geometry prior、teacher guidance 和 curriculum 分别做 ablation。

#### Scenario: loss 关闭回退 baseline CE
- **WHEN** `loss.dba_aware.enabled=false` 或未声明 DBA-aware loss
- **THEN** 系统 MUST 使用现有 hard-label CE 或当前配置声明的 primary beam loss
- **AND** logs MUST 不生成 DBA-aware loss 字段

#### Scenario: ablation metadata 记录
- **WHEN** DBA-aware loss 与 teacher guidance 或 geometry-prior fusion 同时启用
- **THEN** metadata MUST 分别记录每个 objective 的 enabled、weight、temperature/sigma 和 sample count
- **AND** diagnostics MUST 能区分主 beam loss、DBA-aware loss 和 teacher-guidance loss
