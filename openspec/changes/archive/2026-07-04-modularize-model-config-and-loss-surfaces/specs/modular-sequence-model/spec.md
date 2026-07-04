## ADDED Requirements

### Requirement: ModularSequenceModel forward stage 必须可拆分且行为兼容
ModularSequenceModel 实现 MUST 允许 encoder/projector、core input assembly、core/head execution、logit post-processing、geometry/reranker attachment 和 runtime/auxiliary output assembly 位于窄 helper 中，同时不改变 public forward output。

#### Scenario: forward 输出兼容
- **WHEN** ModularSequenceModel internals are refactored
- **THEN** logits, auxiliary outputs, runtime metadata, geometry prior outputs, reranker outputs and `training_strategy_metadata()` MUST remain compatible
- **AND** synthetic forward tests MUST 覆盖 ordinary 和 opt-in metadata 路径
