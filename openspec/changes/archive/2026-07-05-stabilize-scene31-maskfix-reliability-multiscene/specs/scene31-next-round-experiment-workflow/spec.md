## ADDED Requirements

### Requirement: Scene31 subset summary prefers modular maskfix results
Scene31 subset reliability summary MUST prefer `fresh_eval_maskfix/` for AMR-lite and AMBER-lite runs and MUST exclude modular-lite rows without valid maskfix evidence from official winner ranking.

#### Scenario: maskfix result is preferred
- **WHEN** summary reads an AMR-lite or AMBER-lite run with both `fresh_eval_maskfix/` and `fresh_eval/`
- **THEN** it MUST read metrics and mask status from `fresh_eval_maskfix/`
- **AND** output rows MUST include `maskfix_eval`, `mask_suspect`, `excluded_from_official_ranking` and `mask_suspect_reason`

#### Scenario: old eval fallback is excluded
- **WHEN** summary reads an AMR-lite or AMBER-lite run without `fresh_eval_maskfix/`
- **THEN** it MAY fallback to old `fresh_eval/` for visibility
- **AND** it MUST set `mask_suspect=true`, `mask_suspect_reason=no_fresh_eval_maskfix` and `excluded_from_official_ranking=true`

#### Scenario: suspect rows do not rank
- **WHEN** summary generates official ranking, promotion labels or winner conclusions
- **THEN** rows with `mask_suspect=true` or `excluded_from_official_ranking=true` MUST be omitted from the official ranking candidate set
- **AND** they MAY be listed separately as excluded external baselines

#### Scenario: mask status is printed
- **WHEN** summary completes
- **THEN** console output and `combined_conclusion.txt` MUST include AMR/AMBER-lite mask status, whether `fresh_eval_maskfix/` exists, suspect state and ranking inclusion state

### Requirement: Scene31 reliability seed continuation runner
Scene31 subset reliability runner MUST provide focused groups for reliability fusion seed3 and explicitly gated seed4/5 without enabling unrelated methods.

#### Scenario: seed3 group
- **WHEN** the user runs `scripts/run_scene31_subset_reliability.sh --group reliability_seed3 --auto-eval`
- **THEN** the runner MUST select only `proto_randomdrop_subset_reliability_fusion_es40_seed3`
- **AND** the config MUST match seed1/2 except for `seed=3`
- **AND** condBTAPA, weakKD, MPDRO, beamsoft, PatternFiLM, AMR and AMBER MUST remain disabled

#### Scenario: failed seed3 can be overwritten
- **WHEN** seed3 has a failed run directory and the user passes `--overwrite-failed`
- **THEN** the runner MAY replace the failed attempt
- **AND** complete run directories MUST still be skipped unless an explicit overwrite flag is provided

#### Scenario: seed3 auto eval
- **WHEN** seed3 training completes and `--auto-eval` is set
- **THEN** the runner MUST run full fresh eval with the best checkpoint
- **AND** it MUST NOT pass `--max-batches`

#### Scenario: seed4 and seed5 are explicit only
- **WHEN** the user runs `scripts/run_scene31_subset_reliability.sh --group reliability_seed45 --auto-eval`
- **THEN** the runner MUST select seed4 and seed5 reliability fusion configs
- **AND** default `all_new` MUST NOT include seed4 or seed5

### Requirement: Scene31 reliability promotion status
Scene31 combined summary MUST compute reliability fusion status against `proto_randomdrop_subset_es40` after seed3 completes.

#### Scenario: candidate continue gate
- **WHEN** reliability fusion has at least three non-suspect successful seeds
- **THEN** summary MUST compare avg_missing_top1, overall_mean_top1, full_top1 and avg_missing_MAE against `proto_randomdrop_subset_es40`
- **AND** it MUST label the method `candidate_continue_to_seed5` only if avg_missing improves, overall is at least reference, full is within 0.005 below reference and MAE is no worse

#### Scenario: conservative failure label
- **WHEN** the seed3-expanded mean does not satisfy the continue gate
- **THEN** summary MUST label reliability fusion `do_not_expand_now`
- **AND** if miss3 remains lower it MUST mention that caveat in the conclusion

#### Scenario: final combined conclusion fields
- **WHEN** `combined_conclusion.txt` is written
- **THEN** it MUST state the trusted current reference, reliability fusion n and status, PatternFiLM do-not-promote status, AMR/AMBER-lite official ranking status and next-step recommendations
