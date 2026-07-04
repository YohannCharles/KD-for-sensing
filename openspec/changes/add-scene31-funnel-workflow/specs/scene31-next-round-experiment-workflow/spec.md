## ADDED Requirements

### Requirement: Scene31 funnel missing bucket summary
Scene31 apples-to-apples fresh eval summary MUST derive missing buckets by missing modality count and retain the legacy `avg_missing` fields for comparison.

#### Scenario: bucket mapping output
- **WHEN** a funnel or BC-style summary reads per-pattern metrics
- **THEN** it MUST write `missing_bucket_mapping.json` with each observed pattern's `missing_count` and `available_modalities`
- **AND** `full` MUST have `missing_count=0`
- **AND** missing patterns MUST be bucketed as miss1, miss2 or miss3 when the modality count is four

#### Scenario: bucket metrics
- **WHEN** summary writes per-run and method-level tables
- **THEN** it MUST include `miss1_top1`, `miss2_top1`, `miss3_top1`, `avg_missing_top1`, matching `within_3` fields, matching `mae` fields, `overall_mean_top1` and `balanced`
- **AND** `avg_missing` MUST exclude `full`
- **AND** empty buckets MUST emit NaN/blank values and warning text rather than crashing

#### Scenario: bucket sorting
- **WHEN** summary ranks exact Top1
- **THEN** it MUST sort by `avg_missing_top1 desc`, `miss2_top1 desc`, `miss3_top1 desc`, then `full_top1 desc`
- **AND** it MUST emit `rank_by_miss1_top1.md`, `rank_by_miss2_top1.md` and `rank_by_miss3_top1.md`

### Requirement: Scene31 missing-aware checkpoint selection
项目 MUST 提供 `scripts/select_missing_aware_checkpoint.py` 作为 local/manual checkpoint selection 工具。该工具 MUST 支持 `best_full_val`、`best_avg_missing_val`、`best_mixed_val` 和 `best_bucket_balanced_val` 规则，并且不得使用 test/fresh-eval 目标集拟合选择规则。

#### Scenario: checkpoint selection outputs
- **WHEN** 用户对一个或多个 run 执行 missing-aware selection
- **THEN** 工具 MUST 输出 `checkpoint_selection_per_epoch.csv` 和 `checkpoint_selection_summary.csv`
- **AND** summary MUST 包含 `run`、`rule`、`selected_epoch`、`full_top1`、`miss1_top1`、`miss2_top1`、`miss3_top1`、`avg_missing_top1` 和 `score`
- **AND** 每个规则 MUST 在 `selected_checkpoints/<rule_name>/best.ckpt` 生成 symlink 或 copy

#### Scenario: missing epoch checkpoints
- **WHEN** run 只有 best checkpoint 或没有历史 epoch checkpoint
- **THEN** 工具 MUST warning 并跳过不可评估 epoch
- **AND** 可用 checkpoint 仍 MUST 出现在输出 CSV 中

### Requirement: Scene31 funnel local/manual matrix and runner
项目 MUST 提供 Scene31 funnel 配置矩阵与 `scripts/run_scene31_funnel.sh` local/manual launcher。默认输出 root MUST 是 `outputs/scene31_funnel_lmdb`，且 MUST 不覆盖已有 Scene31 roots。

#### Scenario: funnel groups
- **WHEN** 用户运行 `bash scripts/run_scene31_funnel.sh --group main|quick|all|selection|mvfr|mild_mpdro --gpus <ids>`
- **THEN** runner MUST 选择对应 run group
- **AND** `main` MUST 包含 checkpoint selection、JTT seed3/4/5、MVFR seed1/2/3 和 mild MP-DRO P0
- **AND** `quick` MUST 包含 pattern logit calibration、modality-bias、pattern FiLM、TTA 和 PBPR fixed quick-screen runs
- **AND** `all` MUST 包含 main、quick 与 mild MP-DRO P1

#### Scenario: runner safety
- **WHEN** runner 执行训练或 eval
- **THEN** 每张 GPU 同时 MUST 只运行一个 train/eval 进程
- **AND** 已完成 run MUST 默认跳过，`--overwrite` MUST 允许重跑
- **AND** 单个 run 失败 MUST 不终止其它 run
- **AND** 最后 MUST 写出 completed、skipped、failed 和 eval_failed 列表

### Requirement: Scene31 funnel summary and conservative conclusion
项目 MUST 提供 `scripts/summarize_scene31_funnel.py` 汇总 funnel 输出，并给出保守晋级判断。

#### Scenario: funnel summary outputs
- **WHEN** 用户运行 funnel summary
- **THEN** 脚本 MUST 输出 `funnel_per_run.csv`、`funnel_method_mean_std.csv`、`funnel_delta_vs_uniform.csv`、rank markdown、`checkpoint_selection_summary.csv` 和 `funnel_conclusion.txt`
- **AND** method 表 MUST 包含 full、miss1、miss2、miss3、avg_missing、within@3、MAE、overall 和 balanced 的 mean/std 字段

#### Scenario: promotion labels
- **WHEN** quick screen 方法满足 uniform reference 或 miss2/miss3/beam proximity 晋级条件
- **THEN** summary MUST 标记 `promote_to_full_seeds`
- **AND** 不满足条件的方法 MUST 标记 `do_not_promote`
- **AND** 只改善 miss2/miss3 或 MAE 但 avg_missing 不升的方法 MUST 标记 `auxiliary_analysis_candidate`

### Requirement: Scene31 mild MP-DRO training logs
U-MaskBeamJEPA opt-in MP-DRO MUST support mild mixed weighting for funnel runs.

#### Scenario: mild MP-DRO config
- **WHEN** 配置启用 `training.mpdro.enabled=true`
- **THEN** `tau`、`lambda_dro`、`warmup_epochs`、`ema_beta`、`detach_weights`、`full_protection` 和 `min_full_weight` MUST be honored
- **AND** group weights after protection MUST be finite and sum to 1

#### Scenario: mild MP-DRO log columns
- **WHEN** MP-DRO epoch 日志写入
- **THEN** CSV MUST 包含 `epoch`、`pattern`、`ema_loss`、`raw_weight`、`protected_weight` 和 `num_batches`
