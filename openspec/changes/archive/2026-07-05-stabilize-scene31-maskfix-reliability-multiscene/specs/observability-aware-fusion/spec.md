## ADDED Requirements

### Requirement: Reliability fusion Scene31 seed extension guard
Proto-compatible reliability mask weighted fusion MUST only be expanded from seed3 to seed4/5 when the Scene31 summary gate remains positive against `proto_randomdrop_subset_es40`.

#### Scenario: seed config equivalence
- **WHEN** reliability fusion seed3, seed4 or seed5 configs are generated
- **THEN** they MUST match seed1/2 on exposure, proto model family, reliability fusion enabled state, mode `mask_weighted` and max epoch 40
- **AND** only the seed and run name MAY differ

#### Scenario: unrelated methods remain disabled
- **WHEN** reliability fusion continuation configs are generated or run
- **THEN** condBTAPA, weakKD, MPDRO, beamsoft, PatternFiLM, AMR and AMBER MUST be disabled
- **AND** no new transformer, imputation module or external dependency MAY be introduced for this continuation

#### Scenario: expand decision is summary-driven
- **WHEN** seed3 summary status is not `candidate_continue_to_seed5`
- **THEN** seed4/5 MUST remain a prepared explicit group only
- **AND** the default runner group MUST NOT run seed4/5 automatically
