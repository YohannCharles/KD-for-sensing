# jepa-gps-shortcut-benchmark Specification

## Purpose

Post-C2 tombstone for the retired JEPA/GPS shortcut benchmark surface.

## Requirements

### Requirement: JEPA/GPS shortcut benchmark is retired
The project MUST NOT expose the former JEPA/GPS shortcut benchmark as a current package CLI, diagnostic config, or claim gate. Historical smoke/benchmark outputs MUST stay local and MUST NOT promote claims.

#### Scenario: benchmark route cannot be used as current evidence
- **WHEN** current docs, OpenSpec specs, pyproject, and tests are scanned
- **THEN** the benchmark MUST be absent or marked retired
- **AND** final C2 / U-MaskBeamJEPA evidence gates MUST be used instead.
