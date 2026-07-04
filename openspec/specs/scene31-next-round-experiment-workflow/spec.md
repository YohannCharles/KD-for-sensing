# scene31-next-round-experiment-workflow Specification

## Purpose
定义 Scene31 next-round/night-grid local/manual 实验配置矩阵、launcher、fresh eval 汇总与 sanity check 边界。该能力是 manifest-backed 本地实验 surface，不是长期 package CLI。
## Requirements
### Requirement: Scene31 next-round 配置矩阵
项目 MUST 提供 Scene31 next-round local/manual 配置矩阵。矩阵 MUST 覆盖 P0 的 es40 seed 复核与 uniform sampler + selective condBTAPA weak_single λ 组合，并 MAY 额外提供 P1 备选配置。配置 MUST 保持 run name、seed、epoch、sampler 和 condBTAPA 字段一致。

#### Scenario: 生成 P0 配置
- **WHEN** 开发者生成 next-round 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_uniform_es40_seed3/4/5`、`proto_condbtapa_weaksingle_lam005_es40_seed3/4/5`、`proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed1/2/3` 和 `proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1/2/3`
- **AND** 每个配置 MUST 设置 `training.epochs=40` 或 `training.max_epochs=40`
- **AND** 每个配置的 `experiment.seed` MUST 与 run name 中的 seed 一致

#### Scenario: selective condBTAPA 组合配置
- **WHEN** run name 包含 `sampler_uniform_condbtapa_weaksingle`
- **THEN** 配置 MUST 同时启用 `missing_pattern_sampler=uniform` 和 `use_pattern_conditional_btapa=true`
- **AND** `btapa_apply_patterns` MUST 只包含 `radar_only` 与 `lidar_only`
- **AND** 配置 MUST 不启用 naive all-pattern BTAPA

#### Scenario: 配置族 lifecycle 可审计
- **WHEN** 开发者审阅 `configs/scene31/night_grid/` 或 `configs/scene31/next_round/`
- **THEN** project surface inventory MUST 将这些路径标记为 Scene31 local/manual manifest family
- **AND** generator、manifest 和 analysis 脚本 MUST 记录 owner、输出边界和不升级为 package CLI 的 caveat
- **AND** 真实训练命令 MUST 使用本地生成的 YAML：`kd-sensing-train --config <generated-yaml>`

### Requirement: Scene31 P0 fresh eval runner
项目 MUST 提供 local/manual P0 fresh eval runner，用于对已完成训练的 12 个 P0 run 逐个加载 best checkpoint 并执行完整 missing-pattern fresh eval。该 runner MUST 复用现有 apples-to-apples evaluation helper，不复制模型加载、DataLoader 或指标计算逻辑；它仍不是 package CLI。

#### Scenario: 执行 P0 fresh eval
- **WHEN** 用户运行 `scripts/run_scene31_p0_fresh_eval.sh --root outputs/scene31_next_round --gpus <ids>`
- **THEN** runner MUST 从 next-round manifest 的 `group=p0` 自动发现 run，manifest 不存在时 MAY 回退到固定 12 个 P0 run 名
- **AND** runner MUST 使用 `best_val_top1` checkpoint policy，不传入 `--max-batches`
- **AND** 每个 run 的输出 MUST 写入独立 fresh eval 目录，每个 run 的日志 MUST 单独保存
- **AND** 已有完整 fresh eval 结果时默认跳过，`--overwrite` MUST 允许重跑
- **AND** 单个 run 失败 MUST 不终止后续 run，最终 MUST 写出 failed run list 并打印 completed/skipped/failed 数量
- **AND** runner MAY 通过显式选项追加 `amr_net_supervised` 与 `amber_full_architecture` local baseline，但不得因默认缺少这些 checkpoint 影响 P0 12-run 统计

### Requirement: Scene31 next-round 汇总
项目 MUST 提供只读汇总脚本，基于 fresh eval 输出生成 per-run CSV、method mean±std CSV、Markdown 表、相对 proto baseline delta 和 filtered/top10 表。

#### Scenario: 汇总核心指标
- **WHEN** 用户运行 next-round 汇总脚本并传入 output root 或 run dirs
- **THEN** 输出 MUST 包含 `full`、`avg_missing`、`missing_gps`、`missing_radar`、`radar_only`、`lidar_only`、`overall_mean` 和 `balanced`
- **AND** 输出 MUST 包含每个核心指标相对 proto baseline 的 delta
- **AND** `overall_mean` MUST 定义为 `mean(full, missing_gps, missing_radar, radar_only, lidar_only)`，不得擅自混入其它 missing pattern
- **AND** method 表和默认 winner selection MUST 按 `avg_missing`、`full`、`overall_mean`、`balanced` 降序排序
- **AND** `balanced` MUST 只作为辅助排序和参考表，不得作为默认 winner 排序第一指标
- **AND** 输出 MUST 写入 ignored `outputs/scene31_next_round/`、`outputs/scene31/analysis/` 或显式本地路径

#### Scenario: filtered 表兜底
- **WHEN** 没有方法同时满足 configured threshold
- **THEN** 汇总脚本 MUST 输出最接近的 top10
- **AND** 每行 MUST 标注未达标条件

#### Scenario: sanity check
- **WHEN** 汇总脚本读取 P0 fresh eval 结果
- **THEN** 脚本 MUST 检查 best checkpoint、完整 fresh eval metrics、非 `--max-batches`、核心 pattern、`avg_missing`、`overall_mean`、method seed 归并和 `lam0025`/`lam005` tag 口径
- **AND** 发现问题 MUST 在 summary 产物中写出 warning，不得 silent fail

### Requirement: Scene31 BC next-round matrix
Scene31 next-round local/manual generator MUST provide BC experiment run names for adaptive sampler, beam-neighborhood loss, combined ablation and label smoothing baseline. These configs MUST build from the current `proto_sampler_uniform_es40` mainline and MUST NOT enable condBTAPA, weakKD or BTAPA tau1.

#### Scenario: adaptive sampler P0 configs
- **WHEN** 开发者生成 Scene31 next-round 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_adaptive_gap_a05_t1_es40_seed1/2/3`
- **AND** 每个配置 MUST 设置 `missing_pattern_sampler=adaptive_pattern`、`adaptive_score_mode=gap_to_full`、`adaptive_alpha=0.5`、`adaptive_temperature=1.0`、`adaptive_ema_beta=0.9`、`adaptive_warmup_epochs=3`、`adaptive_min_prob=0.05` 和 `adaptive_max_prob=0.40`

#### Scenario: beamsoft P0 configs
- **WHEN** 开发者生成 Scene31 next-round 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_uniform_beamsoft_s15_mix05_es40_seed1/2/3`
- **AND** 每个配置 MUST 使用 `missing_pattern_sampler=uniform`
- **AND** 每个配置 MUST 使用 `loss.type=beam_neighborhood_ce`、`sigma=1.5`、`mix_ce=0.5` 和 `circular=true`

#### Scenario: combined P0 configs
- **WHEN** 开发者生成 Scene31 next-round 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_adaptive_gap_a05_t1_beamsoft_s15_mix05_es40_seed1/2/3`
- **AND** 每个配置 MUST 同时启用 adaptive gap sampler 和 beam-neighborhood loss
- **AND** 配置 MUST 不启用 condBTAPA、weakKD 或 maskadapter，除非 base proto 默认已经启用

#### Scenario: label smoothing baseline configs
- **WHEN** 开发者生成 Scene31 next-round 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_uniform_labelsmooth005_es40_seed1/2/3`
- **AND** 每个配置 MUST 使用 `loss.type=label_smoothing_ce` 和 `smoothing=0.05`

#### Scenario: P1 configs
- **WHEN** 开发者生成完整 Scene31 BC 矩阵
- **THEN** manifest MUST 包含 adaptive loss-score P1、adaptive alpha=0.3 P1、beamsoft sigma=1.0 P1 和 beamsoft sigma=2.0 P1 配置
- **AND** 每个 run name 中的 `a05`、`a03`、`t1`、`s10`、`s15`、`s20`、`mix05`、`es40` 和 `seedN` MUST 与实际配置字段一致

### Requirement: Scene31 BC launcher
项目 MUST 提供 `scripts/run_scene31_bc_next.sh` 作为 local/manual launcher，用于按 group 训练、复评或汇总 Scene31 BC 实验。

#### Scenario: group selection
- **WHEN** 用户运行 `bash scripts/run_scene31_bc_next.sh --group b_p0|c_p0|bc_p0|all_p0|all --gpu 0`
- **THEN** launcher MUST 选择对应 run group
- **AND** `all_p0` MUST 等于 `b_p0 + c_p0 + bc_p0`
- **AND** `all` MUST 包含 `all_p0` 和 P1 configs

#### Scenario: training and eval modes
- **WHEN** 用户传入 `--train-only`
- **THEN** launcher MUST 只训练选中 run
- **AND** 用户传入 `--eval-only` 时 launcher MUST 只执行 fresh eval
- **AND** 未传入二者时 launcher MAY 训练后执行 fresh eval

#### Scenario: skip overwrite and logging
- **WHEN** run output dir 已有 complete state 且未传入 `--overwrite`
- **THEN** launcher MUST 默认跳过该 run
- **AND** 每个 run MUST 保存 stdout/stderr log
- **AND** 单个 run 失败 MUST 继续后续任务
- **AND** 最后 MUST 打印 completed、skipped 和 failed runs

#### Scenario: baseline training inclusion
- **WHEN** 用户运行 `--group baselines`、`--group all_p0` 或 `--group all`
- **THEN** launcher MUST 包含需要训练的 `amr_net_supervised` 与 `amber_full_architecture` baseline
- **AND** baseline MUST 使用已有 config path，不要求进入 Scene31 generated manifest

### Requirement: Scene31 BC summary
Scene31 BC summary MUST 按 seed 归并 method，输出 core metrics、可用 beam-aware metrics、delta vs proto 和 delta vs current uniform winner，并以 `avg_missing` 为第一排序指标。

#### Scenario: summary columns and sorting
- **WHEN** 用户运行 BC summary 脚本
- **THEN** 输出 MUST 包含 `full`、`avg_missing`、`overall_mean`、`missing_gps`、`missing_radar`、`radar_only`、`lidar_only` 和 `balanced`
- **AND** 若输入 fresh eval 包含 `top3`、`top5`、`within_3` 或 `mae`，summary MUST 保留这些指标
- **AND** method 默认排序 MUST 为 `avg_missing desc`、`full desc`、`overall_mean desc`、`balanced desc`

#### Scenario: delta vs uniform winner
- **WHEN** summary 生成 per-run 或 method 表
- **THEN** 输出 MUST 包含 `delta_vs_uniform_full`、`delta_vs_uniform_avg_missing`、`delta_vs_uniform_overall_mean` 和 `delta_vs_uniform_balanced`
- **AND** uniform reference MUST 使用 `proto_sampler_uniform_es40` 当前 winner 数值：`full=0.4216`、`avg_missing=0.2856`、`overall_mean=0.2784` 和 `balanced=0.3560`

### Requirement: Scene31 magic overnight matrix
项目 MUST 提供 Scene31 magic overnight local/manual 配置矩阵，用于独立运行下一批 missing-modality 候选。矩阵 MUST 默认写入 `configs/scene31/magic_overnight/`，输出 root MUST 默认是 `outputs/scene31_magic_overnight_lmdb`，且 MUST 不覆盖 next-round、BC 或 beamsoft weak 既有结果。

#### Scenario: 生成 magic overnight 核心配置
- **WHEN** 开发者生成 magic overnight 配置矩阵
- **THEN** manifest MUST 包含 `proto_sampler_uniform_es40_seed1/2`、`proto_sampler_uniform_mpfr_es40_seed1/2/3`、`proto_uniform_pattern_proto_recenter_es40_seed1/2/3` 和 `proto_uniform_mpdro_tau1_es40_seed1/2/3`
- **AND** 每个配置 MUST 设置 `training.epochs=40` 或 `training.max_epochs=40`
- **AND** 每个配置的 `experiment.seed` MUST 与 run name 中的 seed 一致

#### Scenario: 生成 magic overnight 全量配置
- **WHEN** 开发者请求 `overnight_all` 配置
- **THEN** manifest MUST 额外包含 JTT sample replay baseline、last-layer/prototype retrain baseline 和 vanilla GroupDRO baseline 的 seed1/2 配置
- **AND** manifest MUST 为 proxy/minimal 实现写出 method tags，避免把 overnight proxy 误认为最终 strict algorithm

### Requirement: Scene31 magic overnight 4 GPU runner
项目 MUST 提供 `scripts/run_scene31_magic_overnight.sh` 作为 local/manual launcher。该 runner MUST 支持 `--group overnight_core|overnight_all|mpfr|pbpr|mpdro`、`--gpus <ids>`、`--train-only`、`--eval-only`、`--auto-eval`、`--overwrite` 和 `--root <path>`。

#### Scenario: 四卡并行调度
- **WHEN** 用户运行 `bash scripts/run_scene31_magic_overnight.sh --group overnight_all --gpus 4,5,6,7 --auto-eval`
- **THEN** runner MUST 为 GPU 4、5、6、7 各启动一个 worker
- **AND** 每个 worker 同一时刻 MUST 只运行一个 `kd-sensing-train` 或 fresh eval 进程
- **AND** 单个 run 失败 MUST 不终止其它 worker 或后续 run

#### Scenario: 断点续跑和日志
- **WHEN** run 已有 `state=complete` 的 `run_status.json` 且存在 checkpoint
- **THEN** runner MUST 默认跳过训练
- **AND** run 已有完整 `apples_to_apples_metrics.csv` 时 MUST 默认跳过 fresh eval
- **AND** 每个 run MUST 保存独立 `train.log` 和 `eval.log`
- **AND** runner MUST 写出 completed、skipped、failed、eval_failed 列表，其中 failed list MUST 位于 `overnight_failed_runs.txt`

### Requirement: Missing-pattern DRO training
U-MaskBeamJEPA training extension MUST 支持 opt-in missing-pattern DRO。启用后，系统 MUST 按 batch 中的 missing pattern 更新 EMA loss，按 `softmax(ema_loss / tau)` 计算 group weight，并在 warmup 期间使用 uniform group weight。

#### Scenario: MP-DRO 日志
- **WHEN** 配置启用 `training.mpdro.enabled=true`
- **THEN** 每个 epoch MUST 向当前 run 目录写入 `mpdro_group_log.csv`
- **AND** CSV MUST 包含 `epoch`、`pattern`、`ema_loss`、`weight` 和 `num_batches`
- **AND** 训练日志 MUST 打印每个 epoch 的 pattern weight summary

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

### Requirement: Scene31 manifest-backed workflow 必须保持生成与运行分离
Scene31 next-round、BC、beamsoft weak、funnel 和 magic overnight workflow MUST 使用 manifest、generator、template/base config 与 local/manual runner 分离表达。训练命令 MUST 继续通过 `kd-sensing-train --config <generated-yaml>` 执行。

#### Scenario: 生成字段保持一致
- **WHEN** 用户通过 Scene31 generator 生成 YAML
- **THEN** run name、seed、epoch、sampler、loss、missing-pattern evaluation 和 output root MUST 与 manifest 行一致
- **AND** generator sanity tests MUST 覆盖这些字段

#### Scenario: runner 复用业务实现
- **WHEN** Scene31 runner 执行 train 或 fresh eval
- **THEN** runner MUST 调用 `conda run -n kd_mm_beam kd-sensing-train` 或现有 apples-to-apples helper
- **AND** runner MUST 不复制 DataLoader、模型加载、指标计算或 checkpoint selection 的业务逻辑

