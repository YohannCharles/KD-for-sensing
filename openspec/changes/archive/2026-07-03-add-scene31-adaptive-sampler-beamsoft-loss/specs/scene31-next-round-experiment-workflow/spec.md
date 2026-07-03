## ADDED Requirements

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
