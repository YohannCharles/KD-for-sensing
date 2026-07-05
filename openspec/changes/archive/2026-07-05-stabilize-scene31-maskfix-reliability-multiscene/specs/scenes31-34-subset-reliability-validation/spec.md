## ADDED Requirements

### Requirement: Scene31-34 subset reliability runner
The project MUST provide a local/manual Scene31-34 subset reliability runner for minimal multi-scene validation. The runner MUST stay limited to proto natural, proto sampler uniform, proto randomdrop subset and proto randomdrop subset reliability fusion unless a future change expands the scope.

#### Scenario: quick seed1 group
- **WHEN** the user runs `scripts/run_scenes31_34_subset_reliability.sh --group quick_seed1 --scenes 31,32,33,34 --auto-eval`
- **THEN** the runner MUST select only the four seed1 runs named with `scenes31_34`
- **AND** it MUST write train and eval logs under `outputs/scenes31_34_subset_reliability_lmdb` or the user supplied root

#### Scenario: subset versus reliability seed123 group
- **WHEN** the user runs `scripts/run_scenes31_34_subset_reliability.sh --group subset_vs_reliability_seed123 --scenes 31,32,33,34 --auto-eval`
- **THEN** the runner MUST select only randomdrop subset and randomdrop subset reliability fusion seeds 1, 2 and 3
- **AND** seed2/3 MUST be explicit through this group and MUST NOT be part of the default quick screen

#### Scenario: excluded methods
- **WHEN** Scene31-34 validation configs or groups are generated
- **THEN** PatternFiLM, JTT, MVFR, MPDRO, beamsoft, condBTAPA, weakKD, AMR and AMBER MUST NOT be included

### Requirement: Scene31-34 summary outputs
The project MUST provide a Scene31-34 subset reliability summary script that writes pooled, per-scene and stability artifacts.

#### Scenario: summary output files
- **WHEN** the user runs `scripts/summarize_scenes31_34_subset_reliability.py --root <root> --out <summary_dir>`
- **THEN** the script MUST write `per_run.csv`, `per_scene_method_mean_std.csv`, `pooled_method_mean_std.csv`, `delta_vs_scenes31_34_randomdrop_subset.csv`, `rank_by_avg_missing_top1.md`, `rank_by_scene_stability.md` and `scenes31_34_conclusion.txt`

#### Scenario: stability ranking
- **WHEN** `rank_by_scene_stability.md` is generated
- **THEN** methods MUST be sorted by avg_missing_top1_mean_over_scenes descending, avg_missing_top1_std_over_scenes ascending and avg_missing_MAE_mean_over_scenes ascending

#### Scenario: conservative conclusion
- **WHEN** `scenes31_34_conclusion.txt` is written
- **THEN** it MUST report data/config availability, completed runs, missing or eval failures, current pooled winner, per-scene stability, whether reliability fusion improves over randomdrop subset and whether to expand to seed2/3
