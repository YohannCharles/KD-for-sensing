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

