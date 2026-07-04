## ADDED Requirements

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
