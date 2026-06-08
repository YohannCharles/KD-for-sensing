# gps-pseudo-label-bgam Specification

## Purpose
定义 GPS pseudo-history label 与 BGAM reranker 的因果输入契约，确保历史 pseudo beam、GPS prior、TopK candidates、LiDAR feature、label-space metadata 和评估诊断都可审计且不泄漏 target/query 真实 beam。该能力服务于 MMW/DeepSense 后续 GPS 引导的 LiDAR 注意力与候选重排实验。
## Requirements
### Requirement: 历史 GPS pseudo label 生成
系统 MUST 提供 GPS pseudo-history label generator，用于从 frozen GPS v2 logits/probs、TopK candidate manifest 或配置允许的 geometry fallback 中生成历史 beam/coarse label 序列。pseudo label MUST 使用预测时可观测的 GPS/pose、历史 timestamp、RSU pose、GPS v2 frozen prior 和合法 calibration metadata，不得使用 target/query 真实 beam label、beam power argmax 或 channel oracle。

#### Scenario: 从 GPS v2 logits 生成历史 pseudo label
- **WHEN** 用户为 MMW Town10 scenes 运行 GPS pseudo-history builder 且 MMW GPS v2 logits/probs 可用
- **THEN** 系统 MUST 根据 logits/probs 为每个历史步生成 `history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy` 和 `history_pseudo_confidence`
- **AND** pseudo beam MUST 来自 GPS v2 top1 或配置指定 top-k 聚合
- **AND** metadata MUST 记录 `pseudo_label_source=gps_v2_logits`、GPS artifact 路径、logits index 路径和 frozen prior 标记

#### Scenario: GPS logits 缺失时使用显式 fallback
- **WHEN** GPS v2 logits/probs 缺失且配置允许 geometry fallback
- **THEN** 系统 MUST 使用 GPS/RSU 几何 AoD 映射生成 pseudo beam 或 coarse group
- **AND** metadata MUST 记录 `pseudo_label_source=geometry_fallback`
- **AND** 系统 MUST NOT 静默把 target label、query label 或 beam power oracle 用作 fallback

#### Scenario: pseudo label 不使用未来真实标签
- **WHEN** 改变同一样本的 `target_label`、`gt_beam` 或 future beam power argmax
- **THEN** 该样本的 `history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy` 和 BGAM mask source MUST 保持不变
- **AND** 测试 MUST 覆盖该 anti-leakage 行为

### Requirement: mapped label-space pseudo-history 契约
GPS pseudo-history、TopK candidates、BGAM mask source、训练 label、prediction 和 metrics MUST 位于同一个 beam label-space。MMW GPS pseudo-label BGAM 主实验默认 MUST 使用 `mapping_enabled` 或等价 calibrated label-space，以避免 raw label 跳变；`mapping_disabled` 只能作为显式对照。

#### Scenario: 默认使用 mapping_enabled
- **WHEN** 用户运行默认 GPS pseudo-label BGAM 配置且未传入 `--label-space`
- **THEN** 系统 MUST 使用 `mapping_enabled`
- **AND** 输出目录 MUST 包含 label-space 片段，例如 `outputs/analysis/mmw_town_gps_lidar_bgam/mapping_enabled/`
- **AND** `run_metadata.json`、manifest metadata、summary 和 predictions MUST 记录 `label_space`、`beam_label_space` 和 `beam_label_mapping_fingerprint`

#### Scenario: 禁止混合 label-space
- **WHEN** TopK manifest、GPS pseudo-history artifact、BGAM manifest 或 checkpoint 的 mapping fingerprint 不一致
- **THEN** 系统 MUST 早失败或跳过对应 ablation
- **AND** 错误或 skipped reason MUST 包含 source label-space、target label-space 和 mapping fingerprint

#### Scenario: mapping_disabled 作为对照
- **WHEN** 用户显式传入 `--label-space mapping_disabled`
- **THEN** 系统 MAY 运行 raw label-space 对照
- **AND** raw 对照输出 MUST 与 `mapping_enabled` 主结果分目录保存
- **AND** comparison/aggregation MUST NOT 将 raw 与 mapped 指标混合为同一主结果

### Requirement: 因果时间对齐
系统 MUST 将历史 GPS pseudo label、LiDAR feature 和通信 beam timeline 做因果对齐。历史窗口内每个输入 token MUST 只使用该 token 时间点之前或该时间点已经可观测的信息，并 MUST 记录对齐策略。

#### Scenario: 历史窗口构造
- **WHEN** 构造预测样本的长度为 `history_len=P` 的 pseudo-history
- **THEN** 系统 MUST 输出 `history_timestamps`、`history_pseudo_beams`、`history_pseudo_probs`、`history_pseudo_entropy` 和 `history_valid_mask`
- **AND** 所有历史 timestamp MUST 小于或等于预测 anchor timestamp
- **AND** 不足历史长度的样本 MUST 使用 padding 和 `history_valid_mask=false` 标记

#### Scenario: 低频 LiDAR/GPS 对齐到高频 timeline
- **WHEN** LiDAR 或 GPS 派生特征采样率低于 beam timeline
- **THEN** 系统 MUST 使用 nearest-past、backward/causal replication 或配置指定等价策略对齐
- **AND** metadata MUST 记录 `history_alignment_policy`、sensor period、beam period、replication ratio 和不可对齐样本数

### Requirement: GPS pseudo-history BGAM 输入
BGAM dataset 和 model MUST 支持以历史 GPS pseudo label 替代历史真实 label。模型 forward 的 BGAM 输入 MUST 只包含 LiDAR feature、当前 GPS prior、历史 pseudo label/prob/entropy、GPS TopK candidate 和 beam angle table，不得包含 oracle history label 或 target label。

#### Scenario: dataset 返回 pseudo-history 字段
- **WHEN** `GPSLidarBGAMDataset` 读取启用 pseudo-history 的 manifest
- **THEN** batch MUST 包含 `history_pseudo_beams: LongTensor [B,P]`
- **AND** batch MUST 包含 `history_pseudo_probs` 或 `history_pseudo_log_probs`
- **AND** batch MUST 包含 `history_pseudo_entropy: FloatTensor [B,P]` 和 `history_valid_mask: BoolTensor [B,P]`
- **AND** batch MUST 包含 label-space metadata，至少包括 `beam_label_space` 和 mapping fingerprint

#### Scenario: BGAM 使用 pseudo-history 生成空间 gate
- **WHEN** BGAM mode 为 `history_pseudo_soft`、`history_pseudo_hard` 或 `history_pseudo_topk_union`
- **THEN** 系统 MUST 将历史 pseudo beam 映射为角度或角度分布
- **AND** mask/gate MUST 由历史 pseudo beam/prob/entropy 和当前 GPS uncertainty 生成
- **AND** 输出 mask MUST 不依赖 `gt_beam`、`target_label` 或 oracle history label

#### Scenario: oracle-history 只作为 upper bound
- **WHEN** 配置启用 `oracle_history_bgam_upper_bound`
- **THEN** 系统 MAY 使用历史真实 label 作为对照上界
- **AND** 输出和 summary MUST 明确标记 `uses_oracle_history_label=true`
- **AND** 该 ablation MUST NOT 作为主方法或默认 checkpoint selection 来源

### Requirement: pseudo-history BGAM 评估产物
系统 MUST 为 GPS pseudo-label BGAM 写出可审计的 metrics、predictions、pseudo-history diagnostics、mask diagnostics、mapping metadata 和与 GPS-only 的对比。所有主误差 MUST 使用 mapped label-space 下的 64-beam circular distance。

#### Scenario: summary 输出 pseudo-history 指标
- **WHEN** GPS pseudo-label BGAM evaluation 完成
- **THEN** summary MUST 包含 Top1、Top3、Top5、Top8、DBA、mean/median circular error、delta vs GPS、sample count、pseudo label accuracy on evaluation-only rows、pseudo entropy summary 和 target-in-candidate rate
- **AND** pseudo label accuracy MUST 只作为诊断，不得反向用于训练或 checkpoint selection

#### Scenario: predictions 输出 pseudo-history 字段
- **WHEN** 系统写出 `predictions.csv`
- **THEN** 每行 MUST 包含 `sample_id`、`scene`、`label_space`、`beam_label_mapping_fingerprint`、`gt_beam`、`gps_top1`、`history_pseudo_top1`、`history_pseudo_entropy_mean`、`pred_beam`、`gps_topk`、`model_topk`、`bgam_mode` 和 `uses_oracle_history_label`

#### Scenario: pseudo-history diagnostics 输出
- **WHEN** 系统写出 pseudo-history diagnostics
- **THEN** 系统 MUST 写出 `pseudo_history_summary.csv` 或等价 JSON
- **AND** diagnostics MUST 按 scene、history step、confidence bucket 和 label-space 统计 pseudo label circular error、coverage 和 entropy
- **AND** diagnostics MUST 标记 target/query label 仅用于 evaluation diagnostics
