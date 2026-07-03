# scene31-next-round-experiment-workflow Specification

## Purpose
定义 Scene31 next-round local/manual 实验配置矩阵、launcher、fresh eval 汇总与 sanity check 边界。

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

### Requirement: Scene31 next-round launcher
项目 MUST 提供本地 shell launcher 调度 next-round manifest。launcher MUST 支持 P0/P1/all 分组、GPU id、dry-run、skip completed、overwrite、训练后 fresh eval 和失败列表输出。

#### Scenario: P0 dry-run
- **WHEN** 用户运行 `bash scripts/run_scene31_next_round.sh --group p0 --gpu 0 --dry-run`
- **THEN** launcher MUST 只打印 P0 训练命令
- **AND** 命令 MUST 使用 `conda run -n kd_mm_beam kd-sensing-train`

#### Scenario: 失败继续
- **WHEN** 某个 next-round run 训练失败
- **THEN** launcher MUST 记录该 run 名称并继续后续 run
- **AND** 结束时 MUST 打印失败列表

### Requirement: Scene31 next-round 汇总
项目 MUST 提供只读汇总脚本，基于 fresh eval 输出生成 per-run CSV、method mean±std CSV、Markdown 表、相对 proto baseline delta 和 filtered/top10 表。

#### Scenario: 汇总核心指标
- **WHEN** 用户运行 next-round 汇总脚本并传入 output root 或 run dirs
- **THEN** 输出 MUST 包含 `full`、`avg_missing`、`missing_gps`、`missing_radar`、`radar_only`、`lidar_only` 和 `balanced`
- **AND** 输出 MUST 包含每个核心指标相对 proto baseline 的 delta
- **AND** method 表 MUST 按 `balanced` 降序排序

#### Scenario: filtered 表兜底
- **WHEN** 没有方法同时满足 configured threshold
- **THEN** 汇总脚本 MUST 输出最接近的 top10
- **AND** 每行 MUST 标注未达标条件
