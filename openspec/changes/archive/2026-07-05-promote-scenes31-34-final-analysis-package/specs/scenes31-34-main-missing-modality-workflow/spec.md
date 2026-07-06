## ADDED Requirements

### Requirement: Scene31-34 final statistical evidence package
项目 MUST 提供 `scripts/significance_tests.py`，只读取已有 Scene31-34 fresh eval、summary 和 external-lite artifacts，输出主方法相对关键 baseline 的 seed-level、bootstrap 和 per-scene 统计证据。

#### Scenario: final significance outputs
- **WHEN** 用户运行 `python scripts/significance_tests.py --root outputs/scenes31_34_main_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --out outputs/scenes31_34_main_lmdb/statistics`
- **THEN** 脚本 MUST 输出 `significance_summary.csv`、`significance_summary.md`、`bootstrap_deltas.csv` 和 `per_scene_deltas.csv`
- **AND** 脚本 MUST 写出 `outputs/paper_tables/scenes31_34_main/table_significance_tests.md`
- **AND** accuracy / within@3 delta MUST 使用 method - baseline，MAE / MAE@75 / Top1 drop delta MUST 使用 baseline - method，使正数始终表示主方法更好
- **AND** AMBER-lite n=1 baseline MUST NOT 触发 seed-level paired test，只允许 bootstrap / delta 证据并写 warning

#### Scenario: significance unit consistency
- **WHEN** `significance_tests.py` 写出 `significance_summary.csv`
- **THEN** accuracy-like metrics MUST expose `mean_method_fraction`、`mean_baseline_fraction`、`seed_mean_delta_fraction`、`seed_mean_delta_pp`、`bootstrap_mean_delta_fraction`、`bootstrap_mean_delta_pp`、`bootstrap_ci_low_fraction`、`bootstrap_ci_high_fraction`、`bootstrap_ci_low_pp` 和 `bootstrap_ci_high_pp`
- **AND** MAE metrics MUST expose `mean_method`、`mean_baseline`、`seed_mean_delta`、`bootstrap_mean_delta`、`bootstrap_ci_low` 和 `bootstrap_ci_high` with `metric_unit=beam_index`
- **AND** seed mean delta MUST NOT be mixed into a generic delta column paired with bootstrap CI
- **AND** bootstrap mean delta MUST lie within its bootstrap CI when finite; otherwise the row MUST add a warning in `notes`

### Requirement: Scene31-34 pattern-level evidence package
项目 MUST 提供 `scripts/export_pattern_heatmap.py`，从已有 pattern metrics 导出 pattern-level heatmap、delta 和 win-count evidence。

#### Scenario: final pattern analysis outputs
- **WHEN** 用户运行 `python scripts/export_pattern_heatmap.py --root outputs/scenes31_34_main_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --out outputs/scenes31_34_main_lmdb/pattern_analysis`
- **THEN** 脚本 MUST 输出 top1/MAE matrix CSV、delta-vs-Bernoulli CSV、delta-vs-classifier-subset CSV、`pattern_win_count_summary.csv` 和 PNG/PDF heatmap/delta figures
- **AND** heatmap columns MUST 按 missing_count 分组排序
- **AND** delta figures MUST use positive values to mean proto random subset is better

#### Scenario: final paper pattern figures
- **WHEN** 用户运行 final pattern analysis
- **THEN** 脚本 MUST also output `fig_pattern_delta_vs_bernoulli_top1_paper.png/pdf`、`fig_pattern_delta_vs_bernoulli_mae_paper.png/pdf` 和 `fig_pattern_heatmap_top1_grouped_paper.png/pdf`
- **AND** grouped heatmap MUST use simplified method labels and visible missing-count grouping

### Requirement: Scene31-34 final error CDF evidence
项目 MUST 提供 `scripts/plot_error_cdf.py`，从已有 per-sample prediction artifacts 导出绝对 beam error CDF 数据和论文图。

#### Scenario: final error CDF outputs
- **WHEN** 用户运行 `python scripts/plot_error_cdf.py --root outputs/scenes31_34_main_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --out outputs/scenes31_34_main_lmdb/error_cdf`
- **THEN** 脚本 MUST 输出 `abs_error_cdf_data.csv`
- **AND** 脚本 MUST 输出 `fig_abs_error_cdf_all_missing.png/pdf`、`fig_abs_error_cdf_miss3.png/pdf` 和 `fig_abs_error_cdf_missing_ratio_75.png/pdf`
- **AND** 每张图 MUST 标注 Within@3 threshold

#### Scenario: final paper CDF figures
- **WHEN** 用户运行 error CDF analysis
- **THEN** 脚本 MUST also output `fig_abs_error_cdf_all_missing_paper.png/pdf` and `fig_abs_error_cdf_miss3_paper.png/pdf`
- **AND** caption notes MUST state that higher and left-shifted CDF curves are better

### Requirement: Scene31-34 sampling distribution explanation
项目 MUST 提供 `scripts/summarize_sampling_distribution.py`，解释 Natural、Uniform pattern exposure、Bernoulli randomdrop k075 和 random non-empty subset exposure 的训练采样分布差异。

#### Scenario: final sampling distribution outputs
- **WHEN** 用户运行 `python scripts/summarize_sampling_distribution.py --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --out outputs/scenes31_34_main_lmdb/sampling_analysis`
- **THEN** 脚本 MUST 输出 `sampling_distribution_by_missing_count.csv`、`sampling_distribution_by_subset.csv`、`sampling_distribution_summary.md` 和 PNG/PDF distribution figures
- **AND** 有实际 train log sampling stats 时 MUST 标记 `distribution_source=empirical_from_train_log`
- **AND** 没有实际 stats 时 MUST 根据 config rule 写 `distribution_source=theoretical_from_config`
- **AND** summary MUST 说明 random subset exposure covers non-empty modality subsets and, when uniform, uses `P(S)=1/(2^M-1), S != empty`

### Requirement: Scene31-34 final table updater
项目 MUST 提供 `scripts/update_final_paper_tables.py`，整合 summary、statistics、pattern analysis、profile、error CDF 和 sampling analysis 产物，生成最终论文表格与 notes。

#### Scenario: final paper table update outputs
- **WHEN** 用户运行 `python scripts/update_final_paper_tables.py --summary-root outputs/scenes31_34_main_lmdb/summary --statistics-root outputs/scenes31_34_main_lmdb/statistics --pattern-root outputs/scenes31_34_main_lmdb/pattern_analysis --profile-root outputs/scenes31_34_main_lmdb/profile --cdf-root outputs/scenes31_34_main_lmdb/error_cdf --sampling-root outputs/scenes31_34_main_lmdb/sampling_analysis --paper-table-root outputs/paper_tables/scenes31_34_main`
- **THEN** 脚本 MUST 更新或新增 main、ablation、classifier、external、compute-cost、significance、pattern-win-count、sampling-distribution tables
- **AND** `scenes31_34_final_paper_notes.txt` MUST state the final trusted method, subset-vs-Bernoulli conclusion, classifier baseline interpretation, AMR/AMBER-lite status, and no extra inference-time cost
- **AND** mask_suspect=true external rows MUST NOT enter official ranking

#### Scenario: final paper notes are conservative
- **WHEN** final notes are regenerated
- **THEN** notes MUST state that the final trusted method is prototype + random non-empty subset exposure
- **AND** notes MUST NOT claim prototype alone is sufficient
- **AND** notes MUST state random subset exposure is the primary driver, prototype head adds extra gain under subset exposure, AMR/AMBER-lite do not challenge the final method, random subset has no extra inference-time parameters, latency is measured in compute-cost table, and no further model-search experiments are recommended

### Requirement: Scene31-34 final conclusion consumes final analysis
`scripts/write_scenes31_34_main_conclusion.py` MUST read statistics、pattern_analysis、profile、error_cdf 和 sampling_analysis roots when provided, and write a conservative final conclusion.

#### Scenario: final conclusion sections
- **WHEN** 用户运行 final conclusion 脚本 with all final analysis roots
- **THEN** conclusion MUST include Main winner、Statistical evidence、Pattern-level evidence、Error CDF evidence、Sampling distribution explanation、Compute cost evidence、External baseline result and Whether any further experiments are needed
- **AND** conclusion MUST NOT claim prototype alone has large gains when proto natural and classifier natural are close
- **AND** conclusion MUST NOT suggest continuing new module search

### Requirement: Scene31-34 one-shot final analysis runner
项目 MUST provide `scripts/run_final_scene31_34_analysis.sh` to regenerate final analysis artifacts without training.

#### Scenario: one-shot final analysis
- **WHEN** 用户运行 `bash scripts/run_final_scene31_34_analysis.sh --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --paper-table-root outputs/paper_tables/scenes31_34_main`
- **THEN** runner MUST call significance、pattern heatmap、profile、error CDF、sampling distribution、final paper table update and final conclusion scripts in that order
- **AND** runner MUST support `--skip-profile`、`--skip-external` and `--overwrite`
- **AND** runner MUST NOT start training or modify checkpoint artifacts

### Requirement: Scene31-34 final polish runner and latency benchmark
项目 MUST provide a final polish runner and optional latency benchmark for the Scene31-34 paper-ready artifact refresh.

#### Scenario: lightweight latency benchmark
- **WHEN** 用户运行 `python scripts/profile_scenes31_34_methods.py --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --out outputs/scenes31_34_main_lmdb/profile --benchmark-latency --gpus 5 --warmup-batches 5 --benchmark-batches 50`
- **THEN** profile MUST load existing best checkpoints and run fixed eval dataloader forward timing only
- **AND** output MUST include latency per batch、latency per sample、samples per second、GPU peak memory and device
- **AND** failures MUST write NaN plus warning notes rather than crashing
- **AND** `table_compute_cost.md` MUST use columns Method、Family、Params、Model size、Latency / sample、Samples / second、GPU memory and Extra inference cost

#### Scenario: one-shot final polish
- **WHEN** 用户运行 `bash scripts/run_final_scene31_34_polish.sh --root outputs/scenes31_34_main_lmdb --old-root outputs/scenes31_34_subset_reliability_lmdb --classifier-root outputs/scenes31_34_classifier_lmdb --external-root outputs/scenes31_34_external_lite_lmdb --paper-table-root outputs/paper_tables/scenes31_34_main --gpus 5 --benchmark-latency`
- **THEN** runner MUST call significance、profile、degradation curves、pattern heatmap、error CDF、sampling distribution、final paper table update and final conclusion scripts
- **AND** runner MUST support `--skip-latency`、`--skip-profile`、`--skip-plots` and `--overwrite`
- **AND** runner MUST NOT start training or modify checkpoint artifacts
