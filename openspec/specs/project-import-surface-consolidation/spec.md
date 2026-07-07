# project-import-surface-consolidation Specification

## Purpose

Supporting import-boundary governance for the post-C2 source tree.

## Requirements

### Requirement: Internal imports use current owners
Internal source and tests MUST import concrete current owner modules rather than convenience facades. Retired package facades and benchmark wrappers MUST NOT be restored as a way to satisfy old imports.

#### Scenario: retired facade import is rejected
- **WHEN** architecture boundary tests scan internal Python files
- **THEN** imports from retired BeamBench, BEV-Fusion, Vision-Position, JEPA benchmark, distribution-shift, dataset-audit, and legacy package facades MUST fail the guard
- **AND** current imports MUST point to retained U-Mask, MMW/CSI, modular model, config, or diagnostic owners.
