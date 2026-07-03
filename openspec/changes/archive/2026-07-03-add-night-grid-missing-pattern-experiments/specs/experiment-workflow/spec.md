## ADDED Requirements

### Requirement: Scene31 night grid config generation
项目 MUST 提供 `scripts/generate_experiment_grid.py`，从 `configs/scene31/templates/main_v3_proto_es20_base.yaml` 生成 A-F 共 58 个 run 配置，并在 manifest 中加入 6 个 proto/BTAPA reference run，总计 64 个 run。默认 MUST 不覆盖已有配置。

#### Scenario: 生成 manifest
- **WHEN** 用户运行生成脚本并指定 out_dir
- **THEN** 系统 MUST 写出 `experiment_manifest.csv` 和 `experiment_manifest.json`
- **AND** manifest MUST 包含 `run_name,group,config_path,seed,method_tags,expected_epochs,priority`

#### Scenario: 输出路径唯一
- **WHEN** 生成任一 night grid 配置
- **THEN** 配置中的 run name、exp name 或 output_dir MUST 与其它 run 唯一区分

### Requirement: 8 GPU night grid launcher
项目 MUST 提供 `scripts/run_night_grid_8gpu.sh` 从 manifest 调度训练。launcher MUST 默认 `max_parallel=8`、`num_workers=4`，每个训练进程 MUST 只通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 看到一张 GPU，不默认启用 DDP。

#### Scenario: dry run 只打印命令
- **WHEN** 用户传入 `--dry_run`
- **THEN** launcher MUST 只打印训练命令
- **AND** 每条命令 MUST 包含单个 `CUDA_VISIBLE_DEVICES`

#### Scenario: 失败任务记录
- **WHEN** 任一训练任务返回非零 exit code
- **THEN** run name MUST 写入 `outputs/scene31/analysis/night_grid/failed_runs.txt`
- **AND** 完成任务 MUST 写入 `completed_runs.txt`

### Requirement: night grid fresh eval
项目 MUST 提供 `scripts/eval_night_grid.py` 对 manifest 中已完成 run 做 fresh apples-to-apples eval。该脚本 MUST 使用统一 checkpoint resolver 和统一 missing pattern helper，缺失 checkpoint MUST warning 但不中断。

#### Scenario: 输出 pattern metrics
- **WHEN** eval 脚本找到某 run checkpoint
- **THEN** 输出 `night_grid_metrics.csv`、`night_grid_metrics.md` 和 `checkpoint_manifest.json`
- **AND** CSV 行 MUST 包含 run、group、seed、pattern、Top-K、ADBA、MAE、loss、count、checkpoint path 和 checkpoint epoch

### Requirement: night grid analysis
项目 MUST 提供 `scripts/analyze_night_grid.py`，从 fresh eval 指标计算 by-run、by-group、mean/std、delta-vs-proto、top candidates 和 paper observations。排序 MUST 支持 balanced_score，并惩罚相对 proto 损伤 missing_gps、missing_radar 和 full top1 的候选。

#### Scenario: top candidates 输出
- **WHEN** analysis 脚本运行成功
- **THEN** `night_grid_top_candidates.md` MUST 列出 best avg_missing、best radar_only、best lidar_only、best balanced_score、best without hurting missing_gps、best without hurting missing_radar 和 seed3/40 epoch follow-up top3
- **AND** 若提升小于 seed std，报告 MUST 提示谨慎

### Requirement: summary 兼容 night grid
`scripts/summarize_missing_runs.py` MUST 支持 manifest 输入并识别 night grid run 状态。状态 MUST 至少包括 completed、completed_early_stopped、incomplete_has_checkpoint、killed_or_failed 和 missing。

#### Scenario: manifest summary 字段
- **WHEN** 用户传入 night grid manifest 和 expected epochs
- **THEN** summary 输出 MUST 至少包含 `run_name,group,status,best_epoch,final_epoch,best_val_acc,best_val_adba,best_checkpoint,log_path,exit_code`
