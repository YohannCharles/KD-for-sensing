## ADDED Requirements

### Requirement: Scene31-34 validation availability check
Scene31-34 subset reliability validation MUST check DeepSense6G Scene31, Scene32, Scene33 and Scene34 data/config availability before launching training or evaluation.

#### Scenario: all scenes available
- **WHEN** all requested scenes have supported scene descriptors, config resolution and required local paths
- **THEN** the runner MUST record those scenes as available
- **AND** it MAY proceed with pooled or per-scene validation

#### Scenario: scene missing
- **WHEN** a requested scene lacks data, config resolution or required local paths
- **THEN** the runner MUST print and record a warning for that scene
- **AND** it MUST NOT silently treat the missing scene as Scenario31

#### Scenario: output root isolation
- **WHEN** Scene31-34 validation runs
- **THEN** default output root MUST be `outputs/scenes31_34_subset_reliability_lmdb`
- **AND** the workflow MUST NOT write into `outputs/scene31_baseline_pack_lmdb` or `outputs/scene31_subset_reliability_lmdb`

### Requirement: Scene31-34 pooled and per-scene metric support
Scene31-34 validation MUST support pooled metrics and per-scene metrics for the same run set.

#### Scenario: pooled quick validation
- **WHEN** the user runs the `quick_seed1` Scene31-34 group
- **THEN** the workflow MUST prepare pooled Scene31-34 runs for proto natural, proto sampler uniform, proto randomdrop subset and proto randomdrop subset reliability fusion
- **AND** it MUST use seed1 only by default

#### Scenario: per-scene metrics schema
- **WHEN** Scene31-34 fresh eval or summary outputs per-scene rows
- **THEN** each row MUST include scene, method, seed, full_top1, miss1_top1, miss2_top1, miss3_top1, avg_missing_top1, overall_mean_top1, avg_missing_within@3 and avg_missing_MAE
- **AND** missing scene metrics MUST be represented as unavailable with a warning rather than fabricated numeric values
