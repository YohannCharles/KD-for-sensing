## ADDED Requirements

### Requirement: Proto-compatible reliability mask weighted fusion
The system MUST support a lightweight reliability mask weighted fusion option that is compatible with prototype prediction and randomdrop subset training. Missing modalities MUST receive zero weight and available modality weights MUST be normalized over available modalities only.

#### Scenario: missing modality receives zero weight
- **WHEN** reliability fusion receives modality features and an availability mask
- **THEN** every unavailable modality MUST have weight zero within numerical tolerance
- **AND** unavailable modality features MUST NOT contribute to the fused representation

#### Scenario: available weights normalize
- **WHEN** at least one modality is available for a sample
- **THEN** reliability weights over available modalities MUST sum to one within numerical tolerance
- **AND** the fused representation MUST be equivalent to a weighted sum over available modality features

#### Scenario: lightweight implementation boundary
- **WHEN** reliability fusion is enabled for Scene31 subset candidates
- **THEN** the implementation MUST use a small scorer such as pooled feature plus availability or learned modality reliability embeddings
- **AND** it MUST NOT introduce a complex transformer, imputation module or external dependency

#### Scenario: epoch-level reliability log
- **WHEN** training with reliability fusion completes an epoch
- **THEN** the run directory MUST contain or support writing `reliability_weights_epoch.csv`
- **AND** rows MUST include epoch, pattern, modality, mean_weight, std_weight and available_rate
