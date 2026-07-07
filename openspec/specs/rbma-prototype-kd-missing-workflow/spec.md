# rbma-prototype-kd-missing-workflow Specification

## Purpose

Post-C2 tombstone for the retired RBMA/prototype-KD missing-modality workflow.

## Requirements

### Requirement: RBMA/KD workflow remains retired
The project MUST NOT restore old RBMA, prototype-KD, BTAPA, weakKD, tau/seed sweep YAML, scripts, or claim provenance as current surface. U-MaskBeamJEPA branch implementations MAY remain protected inside the current model, but retired workflow configs and runbooks MUST NOT return without a new OpenSpec change.

#### Scenario: retired RBMA config references are historical only
- **WHEN** current docs, specs, tests, and config references are scanned
- **THEN** old RBMA/KD workflow paths MUST be absent or marked retired/historical
- **AND** current missing-modality evidence MUST use final C2 / U-MaskBeamJEPA or protected MMW/CSI workflows.
