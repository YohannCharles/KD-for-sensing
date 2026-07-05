## ADDED Requirements

### Requirement: Scene31 PatternFiLM d8 follow-up workflow
Scene31 funnel local/manual workflow MUST support a focused PatternFiLM d8 follow-up covering seed1-5, fresh eval with miss1/miss2/miss3 buckets, and conservative comparison against uniform reference. This workflow MUST NOT enable condBTAPA, weakKD, MP-DRO, beamsoft, AMBER, transformer or imputation paths.

#### Scenario: PatternFiLM d8 seed matrix
- **WHEN** 开发者生成 Scene31 funnel 配置矩阵
- **THEN** manifest MUST include `proto_sampler_uniform_pattern_film_d8_es40_seed1/2/3/4/5`
- **AND** 每个 d8 配置 MUST set `training.missing_pattern_sampler=uniform`
- **AND** 每个 d8 配置 MUST set `model.primary.pattern_film.enabled=true`, `dim=8`, `init_identity=true` and `apply_at=pre_head`
- **AND** 每个 d8 配置 MUST set `training.epochs=40` or `training.max_epochs=40`
- **AND** 每个 d8 配置的 `experiment.seed` MUST match the seed suffix in the run name

#### Scenario: Fresh eval includes missing-two patterns
- **WHEN** PatternFiLM d8 或 uniform reference run is fresh-evaluated
- **THEN** pattern-wise CSV MUST include all supported miss2 patterns derived from the configured model modalities
- **AND** for four modalities `gps`, `image`, `radar`, `lidar`, miss2 patterns MUST include `missing_gps_image`, `missing_gps_radar`, `missing_gps_lidar`, `missing_image_radar`, `missing_image_lidar` and `missing_radar_lidar`
- **AND** unsupported modality combinations MUST be skipped with a warning rather than crashing
- **AND** fresh eval MUST use the best checkpoint policy and MUST NOT use `--max-batches`

#### Scenario: Missing bucket mapping records modality semantics
- **WHEN** fresh eval or summary writes `missing_bucket_mapping.json`
- **THEN** every observed pattern MUST include `available_modalities`, `missing_modalities` and `missing_count`
- **AND** `full` MUST have `missing_count=0`
- **AND** missing-one patterns MUST have `missing_count=1`
- **AND** missing-two patterns MUST have `missing_count=2`
- **AND** single-modality-only patterns MUST have `missing_count=3` when the modality count is four

#### Scenario: PatternFiLM d8 summary conclusion
- **WHEN** PatternFiLM d8 summary is generated
- **THEN** outputs MUST include per-run CSV, method mean/std CSV, delta vs uniform CSV, rank markdown files, `missing_bucket_mapping.json` and `patternfilm_conclusion.txt`
- **AND** method rows MUST include full, miss1, miss2, miss3, avg_missing, within@3, MAE, overall and balanced mean/std fields
- **AND** PatternFiLM d8 MUST be marked `promote_to_main_candidate` only when `n>=3`, `avg_missing_top1_mean` exceeds uniform, `overall_mean_top1_mean` exceeds uniform and `full_top1_mean>=0.4078`
- **AND** seed1-only gains MUST be marked conservatively rather than promoted
