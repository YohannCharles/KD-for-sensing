## ADDED Requirements

### Requirement: Scene31 subset reference summary
Scene31 missing-modality summaries MUST use `proto_randomdrop_subset_es40` as the default trusted proto reference. `proto_sampler_uniform_es40` MUST remain visible as an ablation but MUST NOT be used as the default delta or winner reference.

#### Scenario: subset reference selection
- **WHEN** subset reference summary reads baseline pack results
- **THEN** it MUST prefer actual fresh eval rows for `proto_randomdrop_subset_es40` when at least three ok runs are available
- **AND** if fewer than three ok runs are available it MUST use the documented fixed fallback values with a warning

#### Scenario: delta columns use subset reference
- **WHEN** summary writes per-run or method-level delta columns
- **THEN** delta columns MUST be relative to `proto_randomdrop_subset_es40`
- **AND** uniform sampler rows MUST be labeled as ablation rather than current reference

#### Scenario: official ranking excludes suspect rows
- **WHEN** ranking markdown or conclusion files are generated
- **THEN** rows with `mask_suspect=true` MUST NOT participate in official winner ranking
- **AND** suspect modular results MAY be listed separately with the exclusion reason

### Requirement: Scene31 subset reliability and PatternFiLM workflow
Scene31 local/manual workflow MUST provide a focused subset-reliability runner for maskfix eval, reliability fusion candidates, and randomdrop subset + PatternFiLM d8 candidates. The workflow MUST remain manifest/config driven and MUST NOT enable unrelated methods.

#### Scenario: maskfix eval group
- **WHEN** user runs `scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix`
- **THEN** the runner MUST only re-evaluate AMR-lite and AMBER-lite complete runs with best checkpoints
- **AND** it MUST report completed, skipped, failed, eval_failed, missing_checkpoint and mask_suspect lists

#### Scenario: reliability group
- **WHEN** user runs the `reliability` group
- **THEN** selected configs MUST include `proto_randomdrop_subset_reliability_fusion_es40_seed1/2/3`
- **AND** configs MUST use proto framework, randomdrop subset exposure, max epoch 40, best checkpoint and reliability fusion
- **AND** configs MUST NOT enable condBTAPA, weakKD, MPDRO, beamsoft or AMBER

#### Scenario: subset PatternFiLM group
- **WHEN** user runs the `subset_film` group
- **THEN** selected configs MUST include `proto_randomdrop_subset_pattern_film_d8_es40_seed1/2/3`
- **AND** configs MUST use randomdrop subset exposure, `pattern_film.dim=8`, `init_identity=true`, `apply_at=pre_head` and max epoch 40
- **AND** configs MUST NOT enable reliability fusion unless a separate explicit combination is selected

#### Scenario: local runner safety
- **WHEN** the subset reliability runner trains or evaluates candidates on multiple GPUs
- **THEN** each GPU worker MUST run at most one train/eval process at a time
- **AND** complete training and ok eval outputs MUST be skipped by default unless overwrite flags are set
- **AND** per-run train/eval logs and failed lists MUST be written under the selected ignored output root

### Requirement: Scene31 subset combined summary
Scene31 subset combined summary MUST read baseline pack proto results, maskfix modular eval results, reliability fusion results and subset PatternFiLM d8 results, then produce conservative rankings and promotion decisions against `proto_randomdrop_subset_es40`.

#### Scenario: combined summary outputs
- **WHEN** `scripts/summarize_scene31_subset_reliability.py` is run
- **THEN** it MUST write per-run CSV, method mean/std CSV, delta vs randomdrop subset CSV, rank markdown files, suspect modular results and combined conclusion
- **AND** method rows MUST include n, full, miss1, miss2, miss3, avg_missing, overall, within@3, MAE, balanced, mask_suspect_count and main_read fields

#### Scenario: promotion criteria
- **WHEN** a method is considered for promotion
- **THEN** it MUST have at least three non-suspect runs
- **AND** avg_missing_top1_mean MUST exceed the subset reference
- **AND** overall_mean_top1_mean MUST be at least the subset reference
- **AND** full_top1_mean MUST be no more than 0.005 below the subset reference
- **AND** avg_missing_MAE_mean MUST be no worse than the subset reference

#### Scenario: auxiliary-only outcomes
- **WHEN** a method improves only one bucket or beam proximity without meeting the full promotion criteria
- **THEN** summary MUST label it `auxiliary_candidate_only`
- **AND** it MUST NOT be promoted to main candidate
