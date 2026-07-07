# modality-visual-diagnostics Specification

## Purpose

Post-C2 tombstone for retired visual diagnostic entrypoints.

## Requirements

### Requirement: Visual diagnostics do not restore deleted CLIs
The project MUST NOT restore repository viewer, JEPA visual analysis, GPS shortcut benchmark, or other retired visual diagnostic CLIs as current entrypoints.

#### Scenario: current diagnostics are bounded
- **WHEN** diagnostics docs and CLI lifecycle are inspected
- **THEN** current diagnostics MUST be limited to retained U-Mask, MMW/CSI, run-index, preview/dashboard, paper-export, cleanup, and surface-doctor workflows
- **AND** retired visual diagnostic commands MUST not have wrappers.
