# conditional-utility-audit Specification

## Purpose
TBD - created by archiving change add-conditional-utility-audit. Update Purpose after archive.
## Requirements
### Requirement: Conditional audit subset registry
系统 MUST 提供 Conditional Utility Audit 使用的统一 subset registry。Registry MUST 使用中心模态契约标准化模态顺序，并 MUST 定义 `all`、`strong_only`、`strong_plus_image`、`strong_plus_radar`、`strong_plus_lidar`、`single_best_mmwave` 和 `weak_only`。

#### Scenario: 返回 Scene32 audit subset
- **WHEN** 审计入口请求 Scene32 conditional utility subsets
- **THEN** 系统 MUST 返回七个规范 subset 名称
- **AND** 每个 subset 的模态列表 MUST 按 `image`、`radar`、`gps`、`lidar`、`mmwave` 顺序排列
- **AND** 输出 metadata MUST 记录 subset 名称、模态列表和 mask

#### Scenario: validator 复用 subset registry
- **WHEN** `evaluation.modality_subsets.subsets` 包含 conditional audit subset 名称
- **THEN** validator MUST 通过统一 registry 解析 subset
- **AND** validator MUST 不在多个内部函数中重复硬编码这些 subset

### Requirement: Conditional audit runner
系统 MUST 提供独立命令入口，使用一个已训练 MARF checkpoint 和测试集 dataloader 运行 Conditional Utility Audit。该入口 MUST 复用现有配置加载、checkpoint 加载、normalization artifact 加载、模型构建和 batch preparation 语义。

#### Scenario: 运行 Scene32 MARF audit
- **WHEN** 用户运行 `conda run -n kd_mm_beam python tools/analysis/run_conditional_utility_audit.py --config configs/analysis/scene32_marf_conditional_utility_audit.yaml`
- **THEN** 系统 MUST 加载配置指定的 MARF checkpoint
- **AND** 系统 MUST 在测试集上评估 configured subsets
- **AND** 系统 MUST 将产物写入 `outputs/scene32/<run_name>/conditional_utility/`

#### Scenario: 非 MARF 或不支持 mask 的模型
- **WHEN** audit runner 收到不支持 `force_modality_mask` 的 fusion 模型
- **THEN** 系统 MUST 抛出清晰错误
- **AND** 错误信息 MUST 说明 Conditional Utility Audit 需要 subset force mask 支持

### Requirement: Per-sample subset predictions
系统 MUST 为每个样本、每个 horizon、每个 subset 输出逐样本 prediction dump。Dump MUST 包含 ground truth、top-k prediction、top-k probability、ground-truth probability、CE、Top1/Top3/Top5 hit、beam distance 和 DBA contribution。

#### Scenario: 写出 subset predictions
- **WHEN** audit runner 完成 subset forward
- **THEN** 系统 MUST 生成 `subset_predictions.parquet` 或 `subset_predictions.csv.gz`
- **AND** 每一行 MUST 表示一个 `sample_id + horizon + subset`
- **AND** 字段 MUST 至少包含 `sample_id`、`dataset_index`、`horizon_idx`、`horizon_name`、`gt_beam`、`subset_name`、`modalities`、`pred_top1`、`pred_top2`、`pred_top3`、`top1_prob`、`top2_prob`、`top3_prob`、`gt_prob`、`ce`、`top1_hit`、`top3_hit`、`top5_hit`、`beam_distance_top1` 和 `dba_score`

#### Scenario: DBA 逐样本语义一致
- **WHEN** 系统从逐样本 `dba_score` 聚合到 horizon 平均值
- **THEN** 聚合结果 MUST 与 `calculate_dba_score()` 对同一 logits 和 labels 的结果一致
- **AND** DBA MUST 使用 Top-3 prediction 和配置中的 `evaluation.dba_delta`

### Requirement: Marginal utility deltas
系统 MUST 以 `strong_only` 为基线，计算每个弱模态的逐样本边际增益。边际增益 MUST 覆盖 CE、Top1、Top3 和 DBA，并 MUST 按 horizon 汇总。

#### Scenario: 计算 strong plus weak delta
- **WHEN** `subset_predictions` 同时包含 `strong_only` 和 `strong_plus_image`
- **THEN** 系统 MUST 为 image 计算 `delta_ce = ce_strong_only - ce_strong_plus`
- **AND** 系统 MUST 计算 `delta_top1`、`delta_top3` 和 `delta_dba`
- **AND** `delta_ce > 0` MUST 表示加入该弱模态后更好

#### Scenario: 写出 per-sample delta
- **WHEN** audit runner 完成 marginal utility 计算
- **THEN** 系统 MUST 生成 `conditional_utility_per_sample_delta.parquet` 或 `conditional_utility_per_sample_delta.csv.gz`
- **AND** 每一行 MUST 表示一个 `sample_id + horizon + weak_modality`

### Requirement: Teacher complementarity audit
系统 MUST 在显式启用 teacher dump 时，使用 teacher registry 严格加载单模态 teacher，并输出 teacher predictions 与 complementarity summary。Teacher forward MUST 在 `torch.no_grad()` 中执行，teacher MUST 为 eval 模式，且参数 MUST 不需要梯度。

#### Scenario: 写出 teacher predictions
- **WHEN** `dump_teacher_predictions` 为 true
- **THEN** 系统 MUST 从配置的 teacher registry 加载 `image`、`radar`、`gps`、`lidar` 和 `mmwave` teacher
- **AND** 每个 teacher logits MUST 规范化为 `[B, 3, 64]`
- **AND** 系统 MUST 生成 `teacher_predictions.parquet` 或 `teacher_predictions.csv.gz`

#### Scenario: 计算 teacher rescue
- **WHEN** `strong_only_top1_hit == 0` 且某个弱模态 teacher 的 `top1_hit == 1`
- **THEN** 系统 MUST 将该行计为 `teacher_rescue_top1`
- **AND** 系统 MUST 输出每个弱模态的 `rescue_rate_given_strong_top1_wrong`、`teacher_gt_prob_advantage_rate` 和 `teacher_ce_better_than_strong_rate`

### Requirement: Subset oracle
系统 MUST 实现 subset oracle。Oracle MUST 在 `strong_only`、`strong_plus_image`、`strong_plus_radar`、`strong_plus_lidar` 和 `all` 中按每个样本、每个 horizon 的最小 CE 选择 subset，并输出 oracle 指标与选择分布。

#### Scenario: 选择 CE 最小 subset
- **WHEN** 一个样本在同一 horizon 上有多个候选 subset 的 CE
- **THEN** 系统 MUST 选择 CE 最小的 subset 作为 `oracle_subset`
- **AND** 系统 MUST 使用该 subset 的 prediction 计算 oracle Top1、Top3、DBA 和 CE

#### Scenario: 写出 oracle summary
- **WHEN** audit runner 完成 oracle 计算
- **THEN** 系统 MUST 生成 `oracle_subset_summary.json`
- **AND** summary MUST 包含 `oracle_gain_vs_strong_only`、`oracle_choice_distribution` 和 `oracle_choice_distribution_by_horizon`

### Requirement: Communication state buckets
系统 MUST 从已有 batch 字段计算通信状态特征，并按验证集分位数生成 bucket。Bucket 报告 MUST 展示 weak modality 在不同通信状态和 horizon 下相对 `strong_only` 的 delta。

#### Scenario: 计算 mmWave 和 GPS 状态特征
- **WHEN** batch 包含 `mmwave` 和 `gps`
- **THEN** 系统 MUST 计算 `mmwave_entropy`、`mmwave_top1_prob`、`mmwave_top1_top2_margin`、`mmwave_peak_sharpness`、`mmwave_total_power`、`mmwave_peak_drift`、`range_to_bs`、`bearing`、`delta_range`、`delta_bearing`、`angular_velocity` 和 `gps_jump_magnitude`
- **AND** GPS bearing MUST 基于 `atan2(sin_theta, cos_theta)` 计算

#### Scenario: 计算 beam transition bucket
- **WHEN** batch 包含 `input_beam` 和 `target_beam`
- **THEN** 系统 MUST 计算每个 horizon 的 beam transition indicator
- **AND** `t+1` MUST 优先比较最后一个历史 input beam 与第一个 future label

#### Scenario: 写出 bucket 报告
- **WHEN** audit runner 完成 bucket 聚合
- **THEN** 系统 MUST 生成 `conditional_utility_by_bucket.csv`
- **AND** 字段 MUST 至少包含 `bucket_feature`、`bucket_name`、`weak_modality`、`horizon_name`、`num_samples`、`strong_only_top1`、`strong_plus_top1`、`delta_top1`、`strong_only_top3`、`strong_plus_top3`、`delta_top3`、`strong_only_dba`、`strong_plus_dba`、`delta_dba`、`mean_delta_ce`、`positive_delta_ce_rate`、`oracle_choice_rate` 和 `teacher_rescue_rate`

### Requirement: Conditional utility summary and diagnosis
系统 MUST 生成总 summary JSON，汇总 aggregate metrics、marginal utility、oracle、teacher complementarity、bucket highlights、metadata 和 diagnosis。Diagnosis MUST 使用配置阈值，不得硬编码在不可覆盖的位置。

#### Scenario: 写出 summary JSON
- **WHEN** audit runner 完成所有审计步骤
- **THEN** 系统 MUST 生成 `conditional_utility_summary.json`
- **AND** summary MUST 包含 `run_name`、`scene`、`num_samples`、`horizons`、`aggregate_metrics`、`marginal_utility_vs_strong_only`、`marginal_utility_by_horizon`、`oracle_subset`、`teacher_complementarity` 和 `diagnosis`

#### Scenario: 标记 conditional useful
- **WHEN** 某个弱模态 overall delta 不为正，但至少一个满足最小样本数的 bucket 达到配置的 `conditional_delta_dba` 或 `mean_delta_ce` 阈值
- **THEN** diagnosis MUST 将该弱模态标记为 `conditionally_useful`
- **AND** diagnosis MUST 记录触发的 bucket、horizon 和指标值

#### Scenario: 标记 representation exists but not exploited
- **WHEN** 某个弱模态 teacher rescue rate 达到配置阈值，但当前 MARF 的 `strong_plus_<modality>` 没有超过 `strong_only`
- **THEN** diagnosis MUST 将该弱模态标记为 `representation_exists_but_not_exploited`
- **AND** summary MUST 保留触发该判断的 teacher rescue 指标

### Requirement: Audit visualizations
系统 MUST 提供单独绘图脚本，从 audit 输出文件生成核心图表。绘图脚本 MUST 支持 horizon 维度，并 MUST 不覆盖已有可视化工具产物。

#### Scenario: 生成 audit figures
- **WHEN** 用户运行 `conda run -n kd_mm_beam python tools/analysis/analyze_conditional_utility.py --input outputs/scene32/<run_name>/conditional_utility`
- **THEN** 系统 MUST 在 `figures/` 下生成 `subset_metrics_bar.png`、`marginal_delta_by_horizon.png`、`oracle_choice_distribution.png`、`teacher_rescue_rate.png`、`delta_ce_histogram_<modality>.png` 和 `bucket_heatmap_delta_dba.png`
- **AND** 图中 MUST 能区分 positive gain 与 negative gain

### Requirement: Non-invasive audit behavior
Conditional Utility Audit MUST 不改变训练主流程、MARF 模型结构、router 输入、loss 配置、encoder 冻结策略或 checkpoint。所有重计算 MUST 只发生在显式 audit 配置或 audit 脚本中。

#### Scenario: 普通训练不启用 audit
- **WHEN** 用户运行现有训练配置且没有启用 conditional utility audit
- **THEN** 系统 MUST 不生成 per-sample prediction dump
- **AND** 系统 MUST 不加载额外 teacher ensemble
- **AND** 系统 MUST 不改变训练 loss 或 forward 次数

#### Scenario: 普通 evaluate 保持兼容
- **WHEN** 用户运行现有 `scripts/evaluate.py` 或 `src/kd_sensing/cli/evaluate.py`
- **THEN** 系统 MUST 继续保存既有 `metrics.json` 和 `test_report.json`
- **AND** 只有显式配置或脚本请求时才 MAY 额外生成 `conditional_utility/` 目录

