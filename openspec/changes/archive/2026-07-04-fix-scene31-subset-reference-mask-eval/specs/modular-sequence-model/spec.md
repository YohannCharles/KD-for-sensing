## ADDED Requirements

### Requirement: Modular forward missing-mask consumption
`ModularSequenceModel` MUST explicitly accept missing-modality forward parameters and MUST apply the resulting availability mask before modality fusion or representation core execution. The forward parameters MUST be optional and backward compatible.

#### Scenario: forward signature keeps missing kwargs
- **WHEN** shared batch runtime filters kwargs by `ModularSequenceModel.forward` signature
- **THEN** `missing_mask`, `missing_modality_metadata`, `available_modalities` and `modality_mask` MUST be accepted by the model
- **AND** existing callers that omit those kwargs MUST continue to run

#### Scenario: missing modalities are masked before fusion
- **WHEN** fresh eval passes a missing mask where one or more configured modalities are unavailable
- **THEN** the corresponding modality features MUST be zeroed, hard-masked, or excluded before fusion/core computation
- **AND** the model MUST NOT merely pass metadata through without changing the fused input

#### Scenario: metadata is preserved for diagnostics
- **WHEN** `missing_modality_metadata` is supplied to modular forward
- **THEN** model outputs or diagnostics MUST retain enough metadata to report pattern, available modalities and missing modalities
- **AND** forward MUST NOT silently drop the metadata before downstream diagnostics can consume it

#### Scenario: debug logging is bounded
- **WHEN** fresh eval or debug mode requests missing-mask diagnostics
- **THEN** the model or eval path MUST emit a bounded message containing pattern, available modalities, missing modalities and applied modalities
- **AND** normal training MUST NOT print this message for every batch by default
