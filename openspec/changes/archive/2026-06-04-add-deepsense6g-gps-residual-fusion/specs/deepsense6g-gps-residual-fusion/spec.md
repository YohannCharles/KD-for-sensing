## ADDED Requirements

### Requirement: DeepSense6G GPS v2 prior anchored residual workflow
系统 MUST 提供显式 opt-in 的 DeepSense6G GPS v2 prior anchored residual correction workflow。该 workflow MUST 默认覆盖 scenario31、scenario32、scenario33 和 scenario34，使用 `mapping_disabled`、`num_beams=64`、`future_beam1` mmWave power argmax label，并默认使用 `support_ratio=0.15`。

#### Scenario: 默认 residual workflow 配置
- **WHEN** 用户运行 DeepSense6G residual fusion 默认配置
- **THEN** 系统 MUST 解析场景为 scenario31-34
- **AND** 系统 MUST 使用 64 beam circular label 语义
- **AND** 系统 MUST 将默认 support ratio 记录为 `0.15`
- **AND** 系统 MUST 将输出写入 `outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled/`

#### Scenario: 主方法不从零预测 beam
- **WHEN** 系统训练 residual fusion 主 ablation
- **THEN** final prediction MUST 以 GPS v2 prior logits 或 fallback prior logits 为 anchor
- **AND** optional modalities MUST 只用于 residual correction、correction gate 或 candidate re-ranking
- **AND** `modality_only` MUST 只作为 ablation 或反例输出，不得标记为主推荐方法

### Requirement: Residual input inspection
系统 MUST 提供 residual input inspection CLI，用于自动发现 GPS v2 sweep 输入产物、检查关键字段并报告 prior 可用性。该 CLI MUST 不训练模型，不修改 GPS v2 结果，不读取 query label 用于 prior 构造。

#### Scenario: 检查 GPS v2 sweep 文件
- **WHEN** 用户传入 GPS v2 sweep root 与 label space
- **THEN** inspection MUST 检查 r05、r10、r15 和 r20 目录
- **AND** inspection MUST 报告每个 support ratio 的 summary、predictions、per-scene metrics、residual probability 和 figures 路径是否存在
- **AND** inspection MUST 打印关键字段覆盖情况和缺失字段列表

#### Scenario: 检查 predictions 必需字段
- **WHEN** inspection 读取 predictions 文件
- **THEN** 系统 MUST 检查 `scene`、`sample_id`、时间戳或 frame id、`target_label`、GPS top1、top-K、circular error、theta、range、E、N、heading、support/query role 和 ablation/best method 字段
- **AND** 字段缺失时 MUST 报告缺失字段和受影响文件

#### Scenario: 报告 GPS prior source
- **WHEN** inspection 未发现 `gps_logits.npy`、`logits.npy`、`pred_logits.npy` 或 `gps_logits_index.csv`
- **THEN** 系统 MUST 报告 logits 不可用
- **AND** 系统 MUST 指出 residual workflow 将使用 `gps_prior_source=fallback_gaussian_from_top1`，除非用户重跑 GPS v2 并保存 logits

### Requirement: Residual manifest
系统 MUST 提供 residual manifest builder，将 GPS v2 predictions、support/query split、GPS context、target label、GPS prior stats 和 optional modality path/feature 合并为每样本一行的 manifest。manifest MUST 支持没有额外模态时运行 GPS context residual baseline。

#### Scenario: 生成 r15 manifest
- **WHEN** 用户运行 residual manifest builder 且 support ratio 为 `0.15`
- **THEN** 系统 MUST 输出 `outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled/manifest/residual_manifest.csv`
- **AND** manifest MUST 包含 scene、sample id、时间戳或 frame id、support/query role、target label、GPS top1/top3/top5、GPS circular error、signed residual、good/bad 标记、theta、range、E、N、heading、prior stats 和 prior source

#### Scenario: manifest 不使用 query label 构造 prior
- **WHEN** GPS logits 缺失且系统构造 fallback Gaussian prior
- **THEN** prior center MUST 来自 GPS top1 prediction
- **AND** prior construction MUST NOT 使用 target label
- **AND** manifest MUST 记录 `gps_prior_source=fallback_gaussian_from_top1`

#### Scenario: optional modality 缺失不阻断 manifest
- **WHEN** image、LiDAR 或 radar path/feature 在某场景不可用
- **THEN** manifest MUST 在对应列写入空值或不可用标记
- **AND** builder MUST 写出 warning 和 modality availability summary
- **AND** GPS context residual baseline MUST 仍可运行

### Requirement: GPSAnchoredResidualFusion model
系统 MUST 提供 `GPSAnchoredResidualFusion` 模型，用于消费 GPS prior logits、GPS context features 和可选 modality features，并输出 anchored final logits 与诊断信息。

#### Scenario: forward 输出形状稳定
- **WHEN** 模型接收 batch size 为 B 的 `gps_prior_logits: [B, 64]` 与 GPS context features
- **THEN** 模型 MUST 输出 `final_logits: [B, 64]`
- **AND** 模型 MUST 输出 `correction_logits: [B, 64]`
- **AND** 模型 MUST 输出 `modality_only_logits: [B, 64]`
- **AND** 模型 MUST 输出 `correction_gate: [B, 1]`
- **AND** diagnostics MUST 包含 correction scale、prior stats 或 enabled modality metadata

#### Scenario: gated correction 使用 GPS prior anchor
- **WHEN** ablation 为 `gps_plus_residual_gated` 或 `gps_plus_residual_gated_anchor`
- **THEN** final logits MUST 使用 `gps_prior_logits + correction_scale * correction_gate * correction_logits`
- **AND** correction scale MUST 为正数
- **AND** correction scale MUST 受配置的最大值约束

### Requirement: Residual fusion losses
系统 MUST 提供 residual fusion loss 组合，包括 final circular soft CE、modality auxiliary CE、gate BCE、good anchor loss 和 correction L2。loss MUST 支持 hard sample weighting，并且 good anchor loss MUST 只作用于 GPS good 样本。

#### Scenario: hard 样本加权 final CE
- **WHEN** 样本的 GPS circular error 大于或等于 `good_error_threshold`
- **THEN** final circular soft CE MUST 按 `1.0 + hard_sample_weight` 或等价配置权重加权
- **AND** 阈值 MUST 默认为 `4`

#### Scenario: gate target 只来自训练标签
- **WHEN** 系统训练 correction gate
- **THEN** gate target MUST 在训练/support 样本上由 `gps_error >= good_error_threshold` 生成
- **AND** target query label MUST NOT 用于 gate target、early stopping 或模型选择

#### Scenario: good anchor loss 保护 GPS good 样本
- **WHEN** 样本满足 `gps_error < good_error_threshold`
- **THEN** good anchor loss MUST 约束 final distribution 接近 GPS prior distribution
- **AND** 样本不满足该条件时 good anchor loss MUST 不参与该样本的损失

### Requirement: Residual training protocols
系统 MUST 支持 `gps_prior_only`、`target_adapt_beambench_residual` 和 `within_scene_residual_upper_bound` 三类协议。主结论 MUST 来自 `target_adapt_beambench_residual`，并与 GPS v2 baseline 对比。

#### Scenario: target adapt residual split
- **WHEN** 系统对某个 target scene 运行 `target_adapt_beambench_residual`
- **THEN** target support MUST 来自 GPS v2 support split
- **AND** target query MUST 只用于最终评估
- **AND** 当 source prior predictions 完整时，source scenes MUST 可用于 residual pretraining
- **AND** run metadata MUST 记录 train mode、source scenes、target scene、support count 和 query count

#### Scenario: source prior 缺失降级
- **WHEN** source scenes 的 GPS prior predictions 不完整
- **THEN** 系统 MUST 降级为 `support_only`
- **AND** summary MUST 记录 `train_mode=support_only` 和降级原因

#### Scenario: within-scene 只作为上界
- **WHEN** 用户运行 `within_scene_residual_upper_bound`
- **THEN** summary MUST 标记该结果为 upper bound 或 sanity protocol
- **AND** comparison report MUST NOT 将该结果作为 target-adapt 主结论

### Requirement: Residual ablation matrix
系统 MUST 支持一组固定 residual ablation，并在 optional modality 不可用时记录跳过原因。

#### Scenario: 默认 ablation 覆盖
- **WHEN** 用户运行默认 residual experiment
- **THEN** summary MUST 至少包含 `gps_prior_only`、`gps_context_only_residual`、`gps_plus_residual_no_gate`、`gps_plus_residual_gated`、`gps_plus_residual_gated_anchor` 和 `gps_topk_rerank`

#### Scenario: optional modality ablation 可用即运行
- **WHEN** image、LiDAR 或 radar 输入可用且配置允许 auto enable
- **THEN** 系统 MUST 运行对应 `image_plus_gps_residual`、`lidar_plus_gps_residual`、`radar_plus_gps_residual` 或 `all_available_modalities_residual` ablation
- **AND** summary MUST 记录实际启用的 modalities

#### Scenario: optional modality ablation 不可用时跳过
- **WHEN** 某 optional modality 缺失或不可稳定读取
- **THEN** 系统 MUST 跳过对应 ablation
- **AND** summary MUST 写入 `skipped_reason`

### Requirement: GPS anchored top-K reranker
系统 MUST 提供 GPS anchored top-K reranker，在 GPS top-K、GPS top1 local circular window 和 optional modality top-M 的候选集合上打分，并报告候选召回。

#### Scenario: candidate set 包含 wrap-around window
- **WHEN** GPS top1 位于 beam 边界附近且 local radius 大于 0
- **THEN** candidate set MUST 包含通过 circular window 得到的 wrap-around beams
- **AND** candidate set 中 beam id MUST 位于 `[0, num_beams)` 范围内

#### Scenario: rerank loss 只在 target 命中候选时计算
- **WHEN** target beam 不在 reranker candidate set 中
- **THEN** rerank loss MUST 跳过该样本
- **AND** full final logits 分支 MUST 继续按配置处理该样本

#### Scenario: 输出 candidate recall
- **WHEN** reranker 完成评估
- **THEN** 系统 MUST 输出 `target_in_gps_top16`、`target_in_local_radius8` 和 `target_in_union_candidates`
- **AND** 系统 MUST 输出 rerank top1/top3 指标

### Requirement: Residual outputs and comparison report
系统 MUST 写出 residual fusion summary、predictions、correction events、candidate recall、figures 和 GPS v2 comparison report。所有 beam error MUST 使用 circular distance。

#### Scenario: summary 对齐 GPS baseline
- **WHEN** residual evaluation 完成
- **THEN** `summary_overall.csv` 和 `summary_by_scene.csv` MUST 包含 residual 指标和 GPS baseline 指标
- **AND** summary MUST 包含 `delta_DBA_vs_gps`、`delta_mean_error_vs_gps`、`good_sample_degradation_rate` 和 `bad_sample_correction_rate`

#### Scenario: good/bad 分组 summary
- **WHEN** residual evaluation 完成
- **THEN** 系统 MUST 写出 `summary_by_gps_good_bad.csv`
- **AND** 该文件 MUST 分别报告 GPS good 样本与 GPS bad 样本的 count、GPS 指标、final 指标、delta 和 correction/degradation rate

#### Scenario: correction events 只记录变化样本
- **WHEN** final top1 与 GPS top1 不一致
- **THEN** 系统 MUST 在 `correction_events.csv` 记录 scene、sample id、target label、GPS top1、final top1、GPS error、final error、improvement、gate、delta 和 good/bad 状态

#### Scenario: comparison report 自动回答诊断问题
- **WHEN** comparison CLI 读取 GPS v2 summary 与 residual summary
- **THEN** 系统 MUST 写出 markdown report
- **AND** report MUST 回答 residual 是否超过 GPS v2 r15、是否接近 r20、提升来自哪些 scene、hard 样本是否修正、good 样本是否被破坏、gate 是否与 hard 样本相关、多模态是否优于 GPS context residual

### Requirement: Residual visualization
系统 MUST 提供 residual visualization CLI，用于从 residual results 生成 ENU scatter、residual histogram、gate diagnostics、good/bad bar plot、label distribution 和可选 modality montage/heatmap。

#### Scenario: 生成 residual figures
- **WHEN** 用户传入 residual results directory
- **THEN** 系统 MUST 将 figures 写入该目录下的 `figures/`
- **AND** figures MUST 至少覆盖 GPS baseline error、final error、improvement、residual before/after、gate vs GPS error、gate vs improvement 和 good/bad subset bar plot

#### Scenario: optional modality sample visualization
- **WHEN** image、LiDAR 或 radar modality 可用
- **THEN** 系统 MUST 为若干 correction 成功/失败样本输出 image montage 或 feature heatmap
- **AND** modality 不可用时系统 MUST 跳过对应图并记录原因
