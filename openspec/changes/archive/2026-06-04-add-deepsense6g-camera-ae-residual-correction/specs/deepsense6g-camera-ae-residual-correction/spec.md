## ADDED Requirements

### Requirement: DeepSense6G camera residual 默认工作流
系统 MUST 提供显式 opt-in 的 DeepSense6G camera residual correction workflow。该 workflow MUST 默认覆盖 scenario31、scenario32、scenario33 和 scenario34，使用 `mapping_disabled`、`num_beams=64`、`future_beam1_argmax` label，并默认使用 `support_ratio=0.15`。

#### Scenario: 默认配置解析
- **WHEN** 用户运行 camera residual 默认配置
- **THEN** 系统 MUST 解析场景为 scenario31-34
- **AND** 系统 MUST 使用 64 beam circular label 语义
- **AND** 系统 MUST 将默认 support ratio 记录为 `0.15`
- **AND** 系统 MUST 将分析输出写入 `outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/`

#### Scenario: GPS v2 prior 冻结
- **WHEN** 系统训练 camera residual 主方法
- **THEN** GPS v2 adapter MUST NOT 被重新训练或重写
- **AND** final prediction MUST 以 GPS v2 logits 或 fallback prior 为 coarse anchor
- **AND** camera feature MUST 只用于 residual delta、correction gate 或 candidate rerank

#### Scenario: 主方法不从零预测 beam
- **WHEN** 系统运行默认推荐 ablation
- **THEN** camera direct 64-class beam prediction MUST NOT 被标记为主推荐方法
- **AND** camera direct prediction MAY 仅作为 diagnostic ablation 写入 summary

### Requirement: Camera residual manifest
系统 MUST 提供 camera residual manifest builder，将 GPS v2 r15 predictions/logits/fallback prior、support/query split、GPS context、target label、image path 和 AE feature metadata 合并为每样本一行的 manifest。

#### Scenario: 生成 r15 camera manifest
- **WHEN** 用户使用 `support_ratio=0.15` 和 `label_space=mapping_disabled` 运行 manifest builder
- **THEN** 系统 MUST 输出 `outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/manifest/camera_residual_manifest.csv`
- **AND** manifest MUST 包含 `scene`、`sample_id`、timestamp 或 frame id、`support_or_query`、`split_role`、`target_label`、GPS top1/top3/top5、GPS circular error、GPS signed residual、GPS good/bad 标记、GPS context 字段、GPS prior source、GPS logits row index、`image_path` 和 `image_exists`

#### Scenario: GPS logits 优先
- **WHEN** GPS v2 r15 目录中存在 `gps_logits.npy`、`logits.npy` 或 `pred_logits.npy` 以及可对齐的 logits index
- **THEN** manifest builder MUST 使用保存的 logits 构造 GPS prior
- **AND** manifest MUST 记录 `gps_prior_source=saved_logits`

#### Scenario: 缺失 logits 时 fallback
- **WHEN** GPS logits 不可用
- **THEN** manifest builder MUST 只使用 `gps_pred_top1` 构造 circular Gaussian prior
- **AND** manifest MUST 记录 `gps_prior_source=fallback_gaussian_from_top1`
- **AND** prior construction MUST NOT 使用 target label

#### Scenario: image 缺失仍生成 manifest
- **WHEN** 某个样本找不到 image 文件
- **THEN** manifest builder MUST 保留该样本
- **AND** manifest MUST 设置 `image_exists=false`
- **AND** 后续 image/AE 训练 MUST 跳过该样本或给出清晰错误，不得静默成功

### Requirement: Camera AE pretraining
系统 MUST 提供 Camera AutoEncoder 预训练 workflow，用于从 DeepSense6G camera image 中学习 frozen AE feature。Camera AE MUST 使用不依赖联网下载权重的 tiny convolutional autoencoder。

#### Scenario: AE 默认训练数据
- **WHEN** 用户运行 Camera AE 训练
- **THEN** 系统 MUST 默认使用 source scenes 的全部可用 image
- **AND** 系统 MUST 默认使用 target scene 的 support image
- **AND** 系统 MUST 默认不使用 target query unlabeled image
- **AND** 系统 MUST NOT 使用 target query label

#### Scenario: AE 输出产物
- **WHEN** Camera AE 训练完成
- **THEN** 系统 MUST 保存 `outputs/training/deepsense6g_camera_ae/r15/mapping_disabled/checkpoints/best.pt`
- **AND** 系统 MUST 保存 train/val loss metrics
- **AND** 系统 MUST 保存若干 reconstruction examples
- **AND** 系统 MUST 支持 early stopping 和 resume

#### Scenario: 没有 image 时失败清晰
- **WHEN** Camera AE 训练集没有任何可用 image
- **THEN** 系统 MUST 抛出包含原因和 manifest 路径的清晰错误
- **AND** 系统 MUST NOT 写出看似成功的 checkpoint

### Requirement: AE feature extraction
系统 MUST 提供 AE feature extraction workflow，从训练好的 Camera AE encoder 导出每个可用 image 的 fixed latent feature，并生成可与 manifest 对齐的 index。

#### Scenario: 导出 AE features
- **WHEN** 用户传入 Camera AE checkpoint 并运行 feature extraction
- **THEN** 系统 MUST 输出 `outputs/features/deepsense6g_camera_ae/r15/mapping_disabled/features.npy`
- **AND** 系统 MUST 输出 `outputs/features/deepsense6g_camera_ae/r15/mapping_disabled/features_index.csv`
- **AND** `features.npy` 第一维 MUST 与 `features_index.csv` 行数一致

#### Scenario: 生成带 AE 的 manifest
- **WHEN** feature extraction 成功
- **THEN** 系统 MUST 生成 `camera_residual_manifest_with_ae.csv`
- **AND** 新 manifest MUST 包含 `ae_feature_row_index` 和 `ae_feature_path`
- **AND** 没有 image 或没有 feature 的样本 MUST 保留并标记 feature 不可用

### Requirement: CameraGPSResidualFusion 模型
系统 MUST 提供 `CameraGPSResidualFusion` 模型，消费 GPS prior logits、GPS pred top1、GPS context 和 Camera AE feature，并输出 local residual delta distribution、gate、corrected beam distribution 和 final logits。

#### Scenario: forward 输出形状
- **WHEN** 模型接收 batch size 为 B 的 `gps_prior_logits: [B, 64]`、`gps_pred_top1: [B]`、`gps_context: [B, D_gps]` 和 `camera_ae_feature: [B, D_img]`
- **THEN** 模型 MUST 输出 `residual_delta_logits: [B, 2R+2]`
- **AND** 模型 MUST 输出 `correction_gate: [B, 1]`
- **AND** 模型 MUST 输出 `p_corr: [B, 64]`
- **AND** 模型 MUST 输出 `final_logits: [B, 64]`
- **AND** diagnostics MUST 包含 gate、prior source 或 residual delta metadata

#### Scenario: local delta 合成 p_corr
- **WHEN** residual delta radius 为 R
- **THEN** local delta classes MUST 映射到 `[-R, R]`
- **AND** 每个 delta 的 beam MUST 由 `(gps_pred_top1 + delta) mod 64` 得到
- **AND** overflow class MUST 按配置均匀分配到 64 beams 或忽略 contribution
- **AND** `p_corr` MUST 为稳定归一化概率分布

#### Scenario: gated final distribution
- **WHEN** 模型计算 final distribution
- **THEN** 系统 MUST 计算 `p_gps=softmax(gps_prior_logits)`
- **AND** 系统 MUST 计算 `gate=sigmoid(gate_logit)`
- **AND** 系统 MUST 计算 `p_final=(1-gate)*p_gps + gate*p_corr`
- **AND** `final_logits` MUST 等价于 `log(p_final + eps)`

#### Scenario: gate 初始偏向 GPS
- **WHEN** 初始化 `CameraGPSResidualFusion`
- **THEN** gate head bias MUST 默认为负值
- **AND** 默认 bias SHOULD 等价于 `-2.0` 或配置声明的值

### Requirement: Camera residual loss
系统 MUST 提供 Camera residual loss 组合，包含 final circular soft CE、residual delta CE、gate BCE、good-anchor KL、可选 aux direct beam CE 和可选 gate entropy regularization。

#### Scenario: final loss 使用 circular labels
- **WHEN** 系统计算 final CE 或 beam distance
- **THEN** 所有 beam distance、soft label 和指标 MUST 使用 circular beam distance
- **AND** GPS hard sample 的 final CE MUST 按配置加权

#### Scenario: residual delta label
- **WHEN** 系统为训练/support 样本构造 residual delta class
- **THEN** residual MUST 来自 `signed_circular_residual(target_label, gps_pred_top1, num_beams=64)`
- **AND** residual class MUST 使用 local delta window 与 overflow class
- **AND** target query label MUST NOT 用于训练 residual delta head

#### Scenario: gate target
- **WHEN** 系统训练 correction gate
- **THEN** gate target MUST 在训练/support 样本上由 `gps_error >= good_error_threshold` 生成
- **AND** `good_error_threshold` MUST 默认为 `4`
- **AND** target query label MUST NOT 用于 gate target、early stopping 或模型选择

#### Scenario: good-anchor 保护 good 样本
- **WHEN** 样本满足 `gps_error < good_error_threshold`
- **THEN** good-anchor loss MUST 约束 final distribution 接近 GPS prior distribution
- **AND** GPS bad 样本 MUST NOT 被 good-anchor loss 约束

### Requirement: Camera residual training protocols and ablations
系统 MUST 支持 source pretrain、target support finetune 和 support-only 等训练模式，并输出固定 ablation matrix。主结论 MUST 来自 target-adapt camera residual protocol。

#### Scenario: target-adapt split 边界
- **WHEN** 系统对某个 target scene 运行 camera residual training
- **THEN** source scenes MAY 用于 residual/gate head pretrain
- **AND** target support MUST 用于 fine-tune 或 support-only training
- **AND** target query MUST 只用于最终 evaluation
- **AND** GPS v2 prior 和 Camera AE encoder MUST 保持 frozen

#### Scenario: 默认 ablation 覆盖
- **WHEN** 用户运行默认 camera residual experiment
- **THEN** summary MUST 至少包含 `gps_prior_only`、`gps_context_only_residual`、`camera_ae_only_direct_beam`、`camera_ae_plus_gps_concat_direct_beam`、`camera_ae_residual_gated`、`camera_ae_residual_gated_anchor` 和 `camera_ae_residual_gated_anchor_source_pretrain`

#### Scenario: gps_prior_only 复现 r15
- **WHEN** 系统运行 `gps_prior_only`
- **THEN** 系统 MUST 直接使用 GPS v2 prior 或 predictions
- **AND** 系统 MUST 在配置容差内复现 r15 baseline 指标或在 report 中记录原因

### Requirement: Beam candidate attention reranker
系统 MUST 提供可选的 `BeamCandidateAttentionReranker` 最小实现，在 GPS candidate beams 上利用 camera representation 做重排，并报告 candidate recall。

#### Scenario: candidate set 构造
- **WHEN** reranker 构造候选集合
- **THEN** candidate set MUST 包含 GPS top-K beams
- **AND** candidate set MUST 包含 GPS top1 的 local circular window
- **AND** beam id MUST 始终位于 `[0, num_beams)` 范围内

#### Scenario: image pseudo token
- **WHEN** AE 只提供全局 feature 而不提供 patch tokens
- **THEN** reranker MUST 能将 AE feature 作为 pseudo image token 使用
- **AND** 后续 patch token 输入 MUST 保持兼容

#### Scenario: candidate recall 输出
- **WHEN** reranker 完成 evaluation
- **THEN** 系统 MUST 输出 `target_in_gps_top16`
- **AND** 系统 MUST 输出 `target_in_local_radius8`
- **AND** 系统 MUST 输出 `target_in_union_candidates`
- **AND** 系统 MUST 输出 rerank top1/top3 指标

### Requirement: Camera residual outputs and visualization
系统 MUST 写出 camera residual summary、predictions、correction events、candidate recall、comparison report 和 figures。所有结果 MUST 与 GPS v2 r15 baseline 对齐比较。

#### Scenario: summary 输出
- **WHEN** camera residual evaluation 完成
- **THEN** 系统 MUST 写出 `summary_overall.csv`
- **AND** 系统 MUST 写出 `summary_by_scene.csv`
- **AND** 系统 MUST 写出 `summary_by_gps_good_bad.csv`
- **AND** summary MUST 包含 GPS baseline 指标、final 指标、delta 指标、good sample degradation rate、bad sample correction rate 和 gate diagnostics

#### Scenario: predictions 与 correction events
- **WHEN** camera residual evaluation 完成
- **THEN** 系统 MUST 写出 `predictions.csv`
- **AND** 系统 MUST 写出只包含 `final_pred_top1 != gps_pred_top1` 样本的 `correction_events.csv`
- **AND** 每行 MUST 包含 scene、sample id、target label、GPS pred、final pred、GPS error、final error、true residual delta、predicted residual delta、gate 和 image/AE metadata

#### Scenario: figures 输出
- **WHEN** 用户运行 camera residual plot CLI
- **THEN** 系统 MUST 将 figures 写入 `outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/figures/`
- **AND** figures MUST 覆盖 ENU scatter、improvement、residual histogram、signed residual、gate vs GPS error、gate vs improvement、good/bad bar plot、label distribution、residual delta confusion matrix 和 image correction montage

#### Scenario: comparison report
- **WHEN** compare CLI 读取 GPS v2 r15/r20 summary 与 camera residual summary
- **THEN** 系统 MUST 写出 markdown report
- **AND** report MUST 回答是否超过 GPS v2 r15、是否接近 r20、hard samples 是否被修正、good samples 是否被破坏、gate 是否有效以及 camera 是否优于 GPS context residual
