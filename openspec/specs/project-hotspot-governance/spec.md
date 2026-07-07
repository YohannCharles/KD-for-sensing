# project-hotspot-governance Specification

## Purpose

Current supporting governance for right-sizing large owners without restoring retired entrypoints.

## Requirements

### Requirement: Hotspot governance follows post-C2 lifecycle
Hotspot governance MUST classify large owners by current/supporting/retired lifecycle, focused validation, and owner responsibility. It MUST NOT require retired JEPA, BeamBench, BEV-Fusion, Vision-Position, or one-shot script facades to remain.

#### Scenario: large owner is registered without expanding public surface
- **WHEN** a large source owner is retained
- **THEN** inventory MUST document action and focused validation
- **AND** the owner MUST not recreate retired public CLI, wrapper, or package facade surface.
