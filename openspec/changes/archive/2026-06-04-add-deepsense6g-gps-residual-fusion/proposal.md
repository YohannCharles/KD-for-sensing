## Why

DeepSense6G scenario31-34 的 GPS v2 adapter 已经证明 15% target support 基本接近 20% support，且约 85% query 样本能达到 `error < 4`；下一步瓶颈不再是从零预测 beam，而是对少量 hard residual 样本做受控纠偏。该 change 引入以 GPS v2 prior 为 coarse anchor 的 residual correction / re-ranking 阶段，使可用模态只负责修正残差、学习 correction gate 和候选 beam 重排，同时避免破坏 already-good GPS 样本。

## What Changes

- 新增 DeepSense6G GPS v2 residual fusion workflow：默认覆盖 scenario31-34、`mapping_disabled`、64 beam circular label、`future_beam1` mmWave power argmax 口径，并默认使用 `support_ratio=0.15`。
- 新增 GPS v2 结果发现与 residual manifest 构建能力，自动读取 r05/r10/r15/r20 sweep 产物，检查 summary、predictions、per-scene metrics、residual probability、figures 和可选 GPS logits。
- 新增 GPS prior 表示契约：优先使用保存的 `[N, 64]` GPS logits / probabilities；缺失时允许用 GPS top1 构造 circular Gaussian prior，但必须标记 `gps_prior_source=fallback_gaussian_from_top1`，且不得使用 target label 构造 prior。
- 新增 signed circular residual、circular shift、circular window 和 GPS good/bad label 工具，用于 residual label、candidate set、gate target 和诊断。
- 新增 `GPSAnchoredResidualFusion` 模型与 residual losses：输出 `final_logits`、`correction_logits`、`modality_only_logits`、`correction_gate`、`correction_strength` 和 diagnostics；默认以 gated anchor 方式在 GPS prior 上叠加 residual correction。
- 新增可选模态 encoder 与自动可用性处理：GPS context baseline 必须可运行；image/LiDAR/radar feature 或 path 可用时启用相应 ablation，不可用时跳过并写入 `skipped_reason`。
- 新增 `GPSAnchoredTopKReranker`，只在 GPS top-K、local circular window 和可选 modality top-M 候选内重排，并报告 candidate recall。
- 新增 residual training/evaluation/plot/compare CLI：训练协议包含 `gps_prior_only`、`target_adapt_beambench_residual`、`within_scene_residual_upper_bound`，主结果必须与 GPS v2 r15 baseline 对齐比较。
- 新增 summary、predictions、correction events、candidate recall、comparison report 和 residual visualization 输出，所有 beam error 使用 circular distance，并显式报告 `delta_DBA_vs_gps`、hard sample correction rate 和 good sample degradation rate。
- 更新 README，新增 “DeepSense6G residual correction after GPS v2” 章节，说明为何以 GPS prior 为 anchor、为何需要 gate / good anchor loss，以及如何运行和判读结果。
- 不新增顶层 `src.*` 运行入口；实现应放入 `src/kd_sensing/` 包内模块，并通过包内 CLI 或 console script 暴露等价命令，遵守现有项目架构边界。

## Capabilities

### New Capabilities

- `deepsense6g-gps-residual-fusion`: 定义 DeepSense6G scenario31-34 的 GPS v2 prior anchored residual correction、optional modality fusion、top-K re-ranking、训练协议、输出产物、可视化、对比报告和 query leakage guard。

### Modified Capabilities

- `gps-coarse-anchor-prediction`: GPS anchor 下游消费契约扩展到 DeepSense6G GPS v2 sweep artifact、GPS logits/probability 保存与 fallback Gaussian prior。
- `geometry-residual-beam-labels`: residual 工具扩展 signed circular residual、circular shift、circular window 和 GPS good/bad label。
- `modality-aware-data-loading`: residual manifest 必须能自动发现 DeepSense6G optional modality path / feature，并在模态缺失时保持 GPS context baseline 可运行。
- `experiment-workflow`: 增加 residual workflow 的目标场景 support/query 训练边界、query label 禁止用于模型选择、输出 summary 和验收命令约束。
- `project-architecture`: 新 residual workflow 的公开入口必须使用包内 CLI 或 console script，不得新增绕过 `kd_sensing` 包结构的顶层 `src.*` 模块。

## Impact

- 源码：`src/kd_sensing/cli/`、`src/kd_sensing/engine/`、`src/kd_sensing/models/`、`src/kd_sensing/models/encoders/`、`src/kd_sensing/losses/`、`src/kd_sensing/evaluation/`、`src/kd_sensing/data/` 或 `src/kd_sensing/utils/` 中新增 residual workflow、prior loading、manifest、metrics、plots 和 comparison helpers。
- 配置：新增 `configs/deepsense6g_residual_fusion.yaml`，默认指向 `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep` 与 `outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled/`。
- 测试：新增 residual manifest、residual fusion model、residual losses、top-K reranker 和 circular utility tests，并保留现有 `tests/test_mmw_town_gps_adapter_v2.py`、`tests/test_circular_metrics.py` 回归。
- 文档：更新 README 和必要的实验/扩展说明，明确主结论必须对比 GPS v2 baseline，且 query label 只可用于最终评价和诊断图。
- 运行产物：新增 outputs/analysis 下的 residual fusion 结果、figures 和 report；不提交本地训练输出、log、checkpoint、dataset 或缓存。
