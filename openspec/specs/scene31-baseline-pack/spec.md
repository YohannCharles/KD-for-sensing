# scene31-baseline-pack Specification

## Purpose
记录 Scene31 baseline pack 的本地手工训练、fresh eval、random modality dropout、AMR/AMBER-lite/FeatureMod-lite 对照与 summary 口径，确保 baseline pack 与后续 missing-modality 主线结果可复现、可隔离且不覆盖其它 Scene31 输出根。

## Requirements
### Requirement: Scene31 baseline pack run matrix
项目 MUST 提供 Scene31 baseline pack local/manual run matrix。矩阵 MUST 至少覆盖 proto natural、proto random dropout、AMR-lite natural 或 randomdrop、AMR-lite pattern-balanced、AMBER-lite natural 或 randomdrop、AMBER-lite pattern-balanced，并 MAY 覆盖 FeatureMod-lite pattern-balanced。

#### Scenario: baseline pack group 定义
- **WHEN** 用户选择 `randomdrop`、`amr_lite`、`amber_lite`、`featuremod`、`all_core` 或 `all` group
- **THEN** runner MUST 选择对应 run 列表
- **AND** `all_core` MUST 等于 `randomdrop + amr_lite + amber_lite`
- **AND** `all` MUST 等于 `all_core + featuremod`

#### Scenario: 输出 root 隔离
- **WHEN** baseline pack 训练、复评或汇总运行
- **THEN** 默认输出 root MUST 是 `outputs/scene31_baseline_pack_lmdb`
- **AND** 系统 MUST NOT 覆盖 `outputs/scene31_next_round`、`outputs/scene31_bc_next_lmdb`、`outputs/scene31_beamsoft_weak_lmdb`、`outputs/scene31_magic_overnight_lmdb` 或 `outputs/scene31_funnel_lmdb`

### Requirement: Random modality dropout training baseline
系统 MUST 支持 random modality dropout 训练 baseline，并 MUST 与 pattern-balanced sampler 保持实现和 metadata 可区分。

#### Scenario: Bernoulli dropout
- **WHEN** 配置启用 `random_modality_dropout.enabled=true` 且 `mode=bernoulli`
- **THEN** 训练 batch/sample MUST 对每个输入 modality 按 `keep_prob` 独立采样是否保留
- **AND** `ensure_at_least_one_modality=true` 时 MUST 保证每个样本至少保留一个 modality

#### Scenario: Random non-empty subset dropout
- **WHEN** 配置启用 `random_modality_dropout.enabled=true` 且 `mode=random_nonempty_subset`
- **THEN** 训练 batch/sample MUST 从所有非空 available modality subset 中随机采样
- **AND** 采样结果 MUST 能覆盖 miss1、miss2 和 miss3 patterns

#### Scenario: dropout 分布日志
- **WHEN** random modality dropout 训练完成一个 epoch
- **THEN** run 目录 MUST 写入或追加 `random_dropout_pattern_stats.csv`
- **AND** CSV MUST 包含 `epoch`、`pattern_or_available_set`、`num_samples`、`fraction` 和 `missing_count`

### Requirement: Scene31 baseline pack runner
项目 MUST 提供 `scripts/run_scene31_baseline_pack.sh` 作为 local/manual runner。该 runner MUST 支持 4 GPU 或 8 GPU 调度，每张 GPU 同一时刻只运行一个 train 或 eval 进程，并复用现有训练与 fresh eval 入口。

#### Scenario: runner 模式
- **WHEN** 用户传入 `--train-only`
- **THEN** runner MUST 只执行训练
- **AND** 用户传入 `--eval-only` 时 runner MUST 只执行 fresh eval
- **AND** 用户传入 `--auto-eval` 时 runner MUST 在训练完成或跳过后执行 fresh eval

#### Scenario: skip 和 overwrite
- **WHEN** run 已完成训练且未传入 `--overwrite`
- **THEN** runner MUST 默认跳过训练
- **AND** run 已有 `status=ok` fresh eval 且未传入 `--overwrite-eval`
- **THEN** runner MUST 默认跳过评估

#### Scenario: 失败不中断
- **WHEN** 单个 train 或 eval 失败
- **THEN** runner MUST 继续后续 run
- **AND** 最终 MUST 输出 completed、skipped、failed、eval_failed、missing_config 和 missing_checkpoint 统计及 failed list

### Requirement: Baseline pack fresh eval 口径
baseline pack 的正式评估 MUST 复用现有 apples-to-apples fresh eval 和 missing bucket summary 管线。

#### Scenario: 正式 fresh eval
- **WHEN** runner 或用户执行 baseline pack fresh eval
- **THEN** 评估 MUST 使用 best checkpoint
- **AND** 输出 MUST 标记 `checkpoint_used=best`、`status=ok`、`missing_config=0` 和 `missing_checkpoint=0`
- **AND** 评估 MUST NOT 使用 `--max-batches`

#### Scenario: fresh eval 指标完整性
- **WHEN** fresh eval 产物被 summary 读取
- **THEN** 产物 MUST 包含 full、miss1、miss2、miss3、avg_missing、overall、within@3、MAE 和 balanced 口径
- **AND** sanity check MUST 验证 miss1/miss2/miss3 非空、full 存在、top3 >= top1、top5 >= top3、within@3 in `[0,1]` 且 MAE >= 0

### Requirement: Baseline pack summary
项目 MUST 提供 `scripts/summarize_scene31_baseline_pack.py`，用于读取本轮 baseline pack、旧 uniform reference 和可用 proto baseline 结果，并输出 per-run、method-level、delta、rank、参数和保守结论。

#### Scenario: summary 输出文件
- **WHEN** 用户运行 baseline pack summary 脚本
- **THEN** 脚本 MUST 输出 `baseline_per_run.csv`、`baseline_method_mean_std.csv`、`baseline_delta_vs_uniform.csv`、`backbone_training_comparison.csv`、`rank_by_avg_missing_top1.md`、`rank_by_miss1_top1.md`、`rank_by_miss2_top1.md`、`rank_by_miss3_top1.md`、`rank_by_beam_proximity.md`、`params_comparison.csv` 和 `baseline_conclusion.txt`

#### Scenario: summary 字段
- **WHEN** summary 写出 method-level 表
- **THEN** 表 MUST 包含 method、model_family、training_strategy、n、full/miss1/miss2/miss3/avg_missing/overall 的 top1 mean/std、avg_missing within@3 mean/std、avg_missing MAE mean/std、balanced mean/std、total_params_mean、trainable_params_mean 和 extra_params_vs_proto
- **AND** `missing_config`、`missing_checkpoint` 和 failed run MUST NOT 参与 mean/std

#### Scenario: ranking 排序
- **WHEN** summary 生成主排序表
- **THEN** 排序 MUST 依次使用 `avg_missing_top1 desc`、`miss2_top1 desc`、`miss3_top1 desc` 和 `full_top1 desc`

#### Scenario: 保守结论
- **WHEN** summary 写出 `baseline_conclusion.txt`
- **THEN** 结论 MUST 报告 fresh eval status、best method、best lightweight method、random dropout 是否匹配 uniform、pattern-balanced exposure 是否跨 backbone 泛化、复杂 baseline 是否超过 proto+uniform、参数效率和每个 baseline 的 promote/do_not_promote 建议
- **AND** 单 seed quick screen MUST 被明确标记，不得当作最终结论
