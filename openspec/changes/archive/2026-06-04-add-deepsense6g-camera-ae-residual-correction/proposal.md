## Why

DeepSense6G scenario31-34 的 GPS v2 r15 已经形成强 coarse beam prior，15% support 结果基本贴近 20% support；下一阶段的价值不在于让 image 从零预测 64 类 beam，而在于用 camera 表征受控修正 GPS hard residual，同时尽量不破坏 GPS already-good 样本。

本 change 在已完成的 GPS-anchored residual workflow 之上，引入 frozen Camera AE feature、local residual delta 分布、correction gate、good-anchor loss 和可选 beam-candidate query attention reranker，形成可审计、无 query leakage 的 camera-assisted residual 实验方案。

## What Changes

- 新增 DeepSense6G camera residual 配置，默认覆盖 scenario31-34、`mapping_disabled`、`num_beams=64`、`support_ratio=0.15`、`future_beam1_argmax`，输出写入 `outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/`。
- 新增 camera residual manifest builder，自动从 GPS v2 r15 结果读取 predictions、top-K、support/query split、GPS error、signed residual、GPS prior logits 或 Gaussian fallback，并自动发现 image path 与 AE feature path。
- 新增 Camera AE pretraining workflow，使用 tiny convolutional autoencoder，不下载 pretrained weights；默认使用 source scenes image 与 target support image，默认不使用 target query unlabeled image。
- 新增 AE feature extraction workflow，导出 `features.npy`、`features_index.csv`，并生成带 `ae_feature_row_index` / `ae_feature_path` 的 manifest。
- 新增 `CameraGPSResidualFusion` 模型：消费 frozen GPS v2 prior、GPS context 与 frozen AE feature，输出 local residual delta logits、correction gate、`p_corr`、`final_logits` 和 diagnostics。
- 新增 local residual delta class 契约：默认 `delta_radius=8`，类别覆盖 `[-8, +8]` 加 overflow；所有 residual、distance、loss 和指标使用 circular beam distance。
- 新增 Camera residual loss：final circular soft CE、residual delta CE、gate BCE、good-anchor KL、可选 aux direct beam CE 与 gate entropy regularization；hard samples 默认加权，good samples 默认被 anchor 保护。
- 新增 target-adapt camera residual 训练协议：GPS v2 prior frozen、Camera AE encoder frozen，主训练只更新 residual/gate head 与 small fusion MLP；query label 只用于最终评价。
- 新增默认 ablation：`gps_prior_only`、`gps_context_only_residual`、`camera_ae_only_direct_beam`、`camera_ae_plus_gps_concat_direct_beam`、`camera_ae_residual_gated`、`camera_ae_residual_gated_anchor`、`camera_ae_residual_gated_anchor_source_pretrain` 和可选 `camera_ae_query_rerank`。
- 新增最小 `BeamCandidateAttentionReranker`，候选集合来自 GPS top-K 与 GPS top1 local circular window，并报告 candidate recall。
- 新增 summary、predictions、correction events、candidate recall、comparison report 与 camera residual figures，显式对齐 GPS v2 r15 baseline，并报告 hard sample correction 与 good sample degradation。
- 不重写 GPS v2 adapter；不把 camera direct 64-class beam prediction 作为主方法；不新增绕过 `src/kd_sensing` 包结构的顶层 `src.*` 入口。

## Capabilities

### New Capabilities

- `deepsense6g-camera-ae-residual-correction`: 定义 DeepSense6G scenario31-34 的 Camera AE pretraining、AE feature manifest、GPS v2 prior frozen residual/gate correction、camera ablation、candidate attention rerank、输出产物与泄漏防护。

### Modified Capabilities

- `geometry-residual-beam-labels`: residual class 行为扩展为 GPS prior 参照下的 signed circular residual、local delta window 和 overflow class。
- `modality-aware-data-loading`: 增加 camera residual manifest / dataset 对 image path、image missing、AE feature path 和 AE feature index 的数据加载契约。
- `experiment-workflow`: 增加 camera residual 分阶段 CLI、训练协议、query label 禁用边界、输出 summary 和验收命令约束。

## Impact

- 源码：新增或扩展 `src/kd_sensing/data/`、`src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/engine/`、`src/kd_sensing/cli/`、`src/kd_sensing/evaluation/` 和 `src/kd_sensing/diagnostics/` 中的 camera AE、feature extraction、residual/gate、rerank、plot 和 compare 实现。
- 配置：新增 `configs/deepsense6g_camera_residual.yaml`，默认指向 GPS v2 support sweep r15 和 camera residual 输出目录。
- CLI：使用 `python -m kd_sensing.cli.*` 或 `kd-sensing-*` console script 暴露等价入口；用户需求中的 `python -m src.*` 入口只作为功能语义参考，不作为实现入口。
- 测试：新增 camera AE、camera residual fusion、residual delta/loss、beam candidate attention、manifest 和 query leakage guard 单测，并保留 GPS v2 与 circular metrics 回归。
- 文档：更新 README 或实验说明，明确 camera 只做 residual/gate/rerank，不作为主方法从零预测 beam。
- 产物：新增 `outputs/analysis/deepsense6g_camera_residual/`、`outputs/training/deepsense6g_camera_ae/`、`outputs/training/deepsense6g_camera_residual/` 和 `outputs/features/deepsense6g_camera_ae/` 下的本地运行产物；这些产物不进入源码变更。
