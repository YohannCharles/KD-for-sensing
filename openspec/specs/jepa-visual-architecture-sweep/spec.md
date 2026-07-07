# jepa-visual-architecture-sweep Specification

## Purpose

Post-C2 tombstone for the retired JEPA visual architecture sweep surface.

## Requirements

### Requirement: JEPA architecture sweep configs stay retired
The project MUST NOT restore retired JEPA visual architecture sweep manifests, current docs, or package entrypoints without a new OpenSpec change.

#### Scenario: sweep config does not return
- **WHEN** tracked configs and diagnostics docs are inspected
- **THEN** retired JEPA sweep configs MUST not be present as current surface
- **AND** any historical mention MUST be clearly marked retired.
