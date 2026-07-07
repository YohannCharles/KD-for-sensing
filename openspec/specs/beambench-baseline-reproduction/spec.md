# beambench-baseline-reproduction Specification

## Purpose

Post-C2 tombstone for the retired BeamBench/Image AE+GPS reproduction surface.

## Requirements

### Requirement: BeamBench reproduction surface is retired
The project MUST NOT expose BeamBench reproduction as a current package CLI, source workflow, tracked config family, or claim provenance. Historical mentions MUST be marked retired or historical and MUST NOT provide current run commands.

#### Scenario: deleted BeamBench entrypoint stays absent
- **WHEN** maintainers inspect current CLI, configs, docs, tests, or registry surface
- **THEN** BeamBench reproduction entrypoints MUST remain absent
- **AND** current workflows MUST point to final C2 / U-MaskBeamJEPA or protected MMW/CSI paths instead.
