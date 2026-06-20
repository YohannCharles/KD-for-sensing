# safe-residual-beam-rerank-fusion Specification

## Purpose
TBD - created by archiving change add-real-perturbation-residual-rerank-fusion. Update Purpose after archive.
## Requirements
### Requirement: Anchor-safe residual reranker
系统 MUST 支持 opt-in safe residual beam reranker。该 reranker MUST 以 anchor logits 为主预测，只在候选 beam 集合内添加有界 residual score 或执行候选重排。

#### Scenario: 构建 reranker component
- **WHEN** 配置声明 `model.primary.reranker.enabled=true` 且选择 safe residual rerank 类型
- **THEN** 系统 MUST 构建 anchor logits provider、candidate builder 和 residual rerank head
- **AND** final output logits MUST 保持 `[B,T,num_classes]` 或现有 engine 可适配形状

#### Scenario: residual 有界
- **WHEN** reranker 输出 residual score
- **THEN** residual magnitude MUST 受配置的 `max_residual_scale`、temperature 或等价约束限制
- **AND** 非候选 beam MUST 不被无界 residual 改写

### Requirement: Candidate beam set
Reranker MUST 从 anchor top-k、geometry prior top-k、anchor 邻域和可选 teacher top-k 构造 candidate beam set。Candidate set MUST 可诊断并支持 mask 写回全 beam space。

#### Scenario: candidate 来源可追溯
- **WHEN** reranker forward 成功
- **THEN** diagnostics MUST 包含 candidate beam ids、candidate source mask、anchor rank、prior rank 和 selected source
- **AND** candidate recall 或 target rank diagnostics MUST 可在训练/评估阶段输出

#### Scenario: candidate set 为空或 prior 不可用
- **WHEN** geometry prior 不可用或 candidate builder 无有效候选
- **THEN** reranker MUST 回退 anchor logits
- **AND** diagnostics MUST 记录 fallback reason

### Requirement: No-regret fallback gate
Safe residual reranker MUST 提供 no-regret fallback gate。Gate MUST 能在 clean/high-anchor-confidence、prior-image disagreement、GPS invalid 或低 confidence 时保持 anchor prediction。

#### Scenario: clean anchor 保护
- **WHEN** image observability 高、anchor confidence 高且 geometry prior 与 anchor 冲突
- **THEN** reranker MUST 能选择 fallback to anchor
- **AND** clean gate diagnostics MUST 记录 fallback rate 和 clean delta

#### Scenario: hard condition 允许 rerank
- **WHEN** image observability 低或 anchor uncertainty 高，并且 geometry prior 低熵且与候选一致
- **THEN** reranker MAY 改变 anchor top prediction
- **AND** diagnostics MUST 记录 change 是否提高 DBA 或 target rank

### Requirement: Reranker training objectives
Reranker training MUST 保留 hard beam supervised objective，并 MAY 增加 candidate CE、pairwise DBA margin 和 no-regret consistency loss。Evaluation MUST 继续使用 hard target Top-K/DBA。

#### Scenario: candidate CE
- **WHEN** target beam 位于 candidate set 内
- **THEN** candidate CE MAY 监督 reranker 在候选内选择 target 或 DBA-near beam
- **AND** loss metadata MUST 记录 candidate coverage 和 skipped samples

#### Scenario: no-regret consistency
- **WHEN** anchor prediction 已正确或 DBA 高于配置阈值
- **THEN** no-regret loss MAY 惩罚 reranker 明显降低 anchor score margin
- **AND** logs MUST 使用 `loss/rerank_*` 或 `loss/no_regret_*` 等非 retired 命名

