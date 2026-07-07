# training-throughput-optimization Specification

## Purpose

Post-C2 tombstone for the retired standalone training-throughput CLI.

## Requirements

### Requirement: Throughput profiling is local-only
The project MUST NOT declare a standalone public training-throughput CLI after post-C2 cleanup. Local profiling MAY be done around current training commands and MUST keep traces under ignored outputs/logs.

#### Scenario: throughput CLI is absent
- **WHEN** pyproject console scripts and CLI help smoke are checked
- **THEN** the retired throughput command MUST not be declared
- **AND** docs MUST not recommend it as a current workflow.
