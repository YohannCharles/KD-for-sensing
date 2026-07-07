# bev-fusion-2604-reproduction Specification

## Purpose

Post-C2 tombstone for the retired BEV-Fusion 2604 reproduction surface.

## Requirements

### Requirement: BEV-Fusion 2604 is not a current model
The project MUST NOT register or document BEV-Fusion 2604 as a current model, config family, package CLI, or claim provenance. Removed registry-name guards MAY remain only to reject stale configs with a clear hint.

#### Scenario: stale BEV model cannot return
- **WHEN** a current model/config/docs scan runs
- **THEN** BEV-Fusion 2604 MUST appear only in retired or removed-guard context
- **AND** no current tracked config MUST require that model.
