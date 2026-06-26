## ADDED Requirements

### Requirement: Difficulty perturbation cache provenance
The difficulty pipeline MAY be materialized into local cache artifacts for benchmark reuse. Cached artifacts MUST record enough provenance to verify that a loaded perturbed batch corresponds to the requested profile, condition, severity, split, seed and sample ids.

#### Scenario: cache payload 包含 replay metadata
- **WHEN** a perturbed batch is written to cache
- **THEN** payload metadata MUST include profile digest or suite identity, condition, severity, split, seed, sample ids, cache schema version and warnings
- **AND** payload MUST not contain modified target labels or source dataset files

#### Scenario: cache mismatch 被拒绝
- **WHEN** a cached perturbation payload is loaded for a different condition, severity, split, seed or sample id set
- **THEN** loader MUST reject the payload
- **AND** error message MUST include the mismatched cache path or key
