## MODIFIED Requirements

### Requirement: Geometry-prior loss logging is non-retired
Current supervised DBA-aware beam loss 与可选 teacher stabilization MUST 使用非 KD 命名，不得复用 retired distillation runtime。已退役 geometry-prior、safe-rerank 和 `loss/geometry_teacher_kl` 不再属于 current logging contract。

#### Scenario: supervised beam smoothing 命名
- **WHEN** 训练使用 circular Gaussian、distance-aware CE、ordinal/EMD-style loss 或 focal circular loss
- **THEN** logs MUST 使用 `loss/beam_dba_aware`、`loss/beam_circular_soft_ce`、`loss/beam_distance_ce` 或等价非 retired 命名
- **AND** logs MUST 不将该 loss 记录为 `loss/distillation` 或旧 `loss/kd_soft_label`

#### Scenario: Current teacher stabilization 命名
- **WHEN** current model 显式使用 teacher logits/probabilities 作为 stabilization
- **THEN** logs MUST 使用 `loss/teacher_guidance` 或 current owner 的非 KD 名称
- **AND** run metadata MUST 标记 teacher guidance 为 opt-in stabilization

### Requirement: DBA-aware loss ablation isolation
DBA-aware loss MUST 可独立开关，并 MUST 支持与 current teacher guidance 和 curriculum 分别做 ablation。它 MUST 不再要求与 geometry-prior fusion 或 safe reranker 组合。

#### Scenario: loss 关闭回退 baseline CE
- **WHEN** `loss.dba_aware.enabled=false` 或未声明 DBA-aware loss
- **THEN** 系统 MUST 使用现有 hard-label CE 或当前配置声明的 primary beam loss
- **AND** logs MUST 不生成 DBA-aware loss 字段

#### Scenario: ablation metadata 记录
- **WHEN** DBA-aware loss 与 current teacher guidance 或 curriculum 同时启用
- **THEN** metadata MUST 分别记录每个 objective 的 enabled、weight、temperature/sigma 和 sample count
- **AND** diagnostics MUST 能区分主 beam loss、DBA-aware loss 和 teacher-guidance loss
