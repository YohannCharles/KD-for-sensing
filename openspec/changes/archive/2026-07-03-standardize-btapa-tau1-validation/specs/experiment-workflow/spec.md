## ADDED Requirements

### Requirement: BTAPA tau1 seed 与 es20 配置族
项目 MUST 提供不覆盖原始 tau1 的 BTAPA tau1 seed2/seed3 配置和 es20 配置族。除 seed、输出路径和 es20 early stopping 字段外，配置 MUST 与 `main_v3_strong_reliability_btapa_tau1.yaml` 保持一致，并 MUST 不启用 RBMA、JEPA、KD、fullaux 或 ADBA-aware proto。

#### Scenario: seed 配置不覆盖原始 run
- **WHEN** 用户运行 `main_v3_strong_reliability_btapa_tau1_seed2.yaml` 或 `main_v3_strong_reliability_btapa_tau1_seed3.yaml`
- **THEN** 输出路径或 run name MUST 包含 `btapa_tau1_seed2` 或 `btapa_tau1_seed3`
- **AND** 配置 MUST 支持 `--auto_resume`

#### Scenario: es20 配置启用短训练早停
- **WHEN** 用户运行任一 `main_v3_strong_reliability_btapa_tau1_es20*.yaml`
- **THEN** 配置 MUST 设置 `max_epochs: 20` 或项目等价字段
- **AND** 配置 MUST 启用以 `val_top1` 或项目等价字段为指标的 early stopping、patience 5 和 best checkpoint 选择

### Requirement: 关键 BTAPA tau1 验证 launcher
项目 MUST 提供只运行关键 BTAPA tau1 验证任务的串行 launcher。launcher 默认 MUST 不并发训练，MUST 支持 dry run、num_workers、max_parallel、gpu_ids、skip_train、skip_eval 和 skip_analysis，并 MUST 不运行 tau4、ADBA、modw1、fusiononly、RBMA、JEPA、KD 或 fullaux。

#### Scenario: dry run 只打印命令
- **WHEN** 用户运行 `bash scripts/run_btapa_tau1_validation.sh --dry_run --num_workers 4 --max_parallel 1`
- **THEN** launcher MUST 只打印 apples-to-apples、tau1 seed/es20 训练、seed 分析和 summary 命令
- **AND** 训练命令 MUST 包含 `--auto_resume`

#### Scenario: 默认串行执行
- **WHEN** 用户不传 `--max_parallel`
- **THEN** launcher MUST 使用 `max_parallel=1`
- **AND** 每个训练任务 MUST 写入独立日志

### Requirement: proto vs BTAPA 8GPU launcher
项目 MUST 提供 `scripts/run_proto_vs_btapa_8gpu.sh`，默认调度 ordinary proto 三 seed 与 BTAPA tau1 三 seed。launcher MUST 默认 `max_parallel=8`、每个训练进程只通过 `CUDA_VISIBLE_DEVICES=<gpu_id>` 看到单张 GPU、默认 `num_workers=4`，并支持 dry-run、skip/only、skip-completed、auto-resume、stagger start 和训练后 fresh eval。

#### Scenario: dry run only-proto
- **WHEN** 用户运行 8GPU launcher 并传入 `--dry_run --only_proto --num_workers 4 --max_parallel 8 --gpu_ids 0,1,2,3,4,5,6,7 --auto_resume --skip_completed`
- **THEN** launcher MUST 只打印 proto 三 seed 的训练命令
- **AND** 每条训练命令 MUST 绑定单个 `CUDA_VISIBLE_DEVICES`

#### Scenario: eval after train
- **WHEN** 用户传入 `--eval_after_train` 或 `--run_eval`
- **THEN** launcher MUST 在训练子进程结束后调用 `scripts/reevaluate_apples_to_apples.py`
- **AND** 缺失 checkpoint 的 run MUST warning 但不阻断其它 run 复评

### Requirement: proto vs BTAPA seed mean±std 分析
项目 MUST 提供 `scripts/analyze_proto_vs_btapa_seeds.py`，读取 fresh apples-to-apples eval 输出，生成 seed metrics、mean±std、delta 和 paper-ready observation。报告 MUST 重点列出 full、avg_missing、missing_gps、radar_only、lidar_only 的 Top-1 和 avg_missing ADBA，并在 delta 小于 std 时提示谨慎报告。

#### Scenario: 输出 mean std 与 delta
- **WHEN** 用户传入 proto 三 seed、BTAPA tau1 三 seed 和 fresh eval 目录
- **THEN** 脚本 MUST 输出 seed metrics、mean±std、delta mean 和 Markdown 报告
- **AND** Markdown MUST 包含保守 paper-ready observation 与 seed 方差提示
