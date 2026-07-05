## ADDED Requirements

### Requirement: Modular-lite missing-mask fresh eval diagnostics
AMR-lite and AMBER-lite local baseline fresh eval MUST verify that missing-pattern masks are received and affect outputs. Results where full and missing outputs or metrics are indistinguishable MUST be marked suspect and excluded from official winner ranking.

#### Scenario: diagnostics report mask path
- **WHEN** `scripts/diagnose_modular_missing_mask.py` is run against a baseline root
- **THEN** it MUST write `modular_missing_mask_diagnostics.csv`
- **AND** each row MUST include model name, run name, forward signature, whether missing-mask kwargs are accepted, whether eval passes a mask, whether batch filtering drops it, whether forward applies it, full-vs-missing equality and a diagnosis

#### Scenario: identical logits warn
- **WHEN** the diagnostic can compare full and missing pattern logits on the same batch
- **THEN** exactly equal full and missing logits MUST produce a warning diagnosis
- **AND** unsupported or unavailable checks MUST be explicit rather than reported as ok

#### Scenario: maskfix fresh eval does not retrain
- **WHEN** the maskfix eval runner processes AMR-lite or AMBER-lite runs
- **THEN** it MUST load the existing best checkpoint for complete runs
- **AND** it MUST NOT start training or overwrite old checkpoint files
- **AND** it MUST NOT pass `--max-batches`

#### Scenario: suspect results excluded
- **WHEN** full and missing pattern metrics remain exactly identical after maskfix fresh eval
- **THEN** the run MUST be marked `mask_suspect=true`
- **AND** summary scripts MUST exclude that run from official winner ranking and promotion decisions
