## ADDED Requirements

### Requirement: Modular-lite formal maskfix fresh eval artifacts
AMR-lite and AMBER-lite formal maskfix evaluation MUST write a separate `fresh_eval_maskfix/` artifact set and MUST NOT overwrite existing `fresh_eval/` results or checkpoints.

#### Scenario: maskfix eval writes required files
- **WHEN** the maskfix eval runner processes a complete AMR-lite or AMBER-lite run with a best checkpoint
- **THEN** it MUST write `fresh_eval_maskfix/apples_to_apples_metrics.csv`, `fresh_eval_maskfix/pattern_metrics.csv`, `fresh_eval_maskfix/mask_suspect.json` and `fresh_eval_maskfix/eval_log.txt`
- **AND** it MUST record `maskfix_eval=true`, run name, method, checkpoint policy and checkpoint path

#### Scenario: maskfix eval skips unavailable runs
- **WHEN** an AMR-lite or AMBER-lite run directory, config or checkpoint is missing
- **THEN** the runner MUST skip that run with a warning
- **AND** it MUST NOT start training or create replacement checkpoints

#### Scenario: old eval is preserved
- **WHEN** `fresh_eval/` already exists for a modular-lite run
- **THEN** maskfix evaluation MUST write to `fresh_eval_maskfix/`
- **AND** it MUST NOT delete, mutate or reinterpret the old `fresh_eval/` directory as maskfix evidence

### Requirement: Modular-lite mask suspect artifact
AMR-lite and AMBER-lite maskfix evaluation MUST automatically mark suspicious results and expose the reason in machine-readable artifacts.

#### Scenario: identical metrics are suspect
- **WHEN** full, missing_gps, radar_only and lidar_only core metrics are exactly identical after maskfix evaluation
- **THEN** `mask_suspect.json` MUST contain `mask_suspect=true`
- **AND** the reason MUST mention identical core metrics

#### Scenario: identical logits are suspect
- **WHEN** the evaluation can compare full logits with missing-pattern logits and they are exactly equal
- **THEN** `mask_suspect.json` MUST contain `logits_full_vs_missing_equal=true`
- **AND** the run MUST be marked suspect

#### Scenario: missing mask application is required
- **WHEN** any evaluated missing pattern has `mask_applied=false`, an incorrect `missing_count`, or `maskfix_eval` is not true
- **THEN** the run MUST be marked suspect
- **AND** the suspect reason MUST be written to `mask_suspect.json`

#### Scenario: non-suspect artifact records checked patterns
- **WHEN** no suspect condition is found
- **THEN** `mask_suspect.json` MUST contain `mask_suspect=false`, an empty reason, checked patterns and identical metric groups
- **AND** the artifact MUST still record whether logits equality was checked or unavailable
