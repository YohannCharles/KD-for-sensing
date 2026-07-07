# jepa-visual-analysis-suite Specification

## Purpose

Post-C2 tombstone for the retired JEPA visual analysis diagnostic surface.

## Requirements

### Requirement: JEPA visual analysis is retired
The project MUST NOT expose the former JEPA visual analysis diagnostic as a current console script, current config, or recommended documentation command. Historical notes MAY remain only with retired context.

#### Scenario: visual diagnostic command is absent
- **WHEN** CLI smoke and project-surface doctor inspect public commands
- **THEN** the retired JEPA visual analysis command MUST not be declared
- **AND** no compatibility wrapper or alias MUST be added.
