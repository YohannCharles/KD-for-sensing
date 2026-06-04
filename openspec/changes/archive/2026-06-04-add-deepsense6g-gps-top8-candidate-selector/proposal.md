## Why

DeepSense6G scenario31-34 的 GPS v2 adapter 已完成 support sweep 与 TopK 分析：`support_ratio=0.15` 已足够作为默认设置，GPS v2 r15 的 Top8 recall 约为 0.8733，Top16 recall 约为 0.9196。下一阶段的核心机会不是让 image、LiDAR 或 radar 从零预测 64 类 beam，也不是优先做 64 类 residual delta correction，而是把 GPS v2 作为强 candidate generator，让其他模态在 GPS Top8 候选内做受 GPS prior 约束的选择与重排。

## What Changes

- 新增 DeepSense6G GPS Top8 Candidate Selector workflow：默认覆盖 scenario31、scenario32、scenario33、scenario34，使用 `mapping_disabled`、`num_beams=64`、`future_beam1` mmWave power argmax label、circular beam distance，并默认使用 `support_ratio=0.15`。
- 新增 GPS v2 Top8 candidate manifest 构建能力：必须从保存的 GPS v2 logits 重新计算 Top8 candidates、GPS score/prob/rank、GPS context、Top8 hit/miss、oracle candidate、candidate rank distribution 和 optional modality 可用性。
- 新增 TopK candidate dataset 契约：返回 `[8]` candidates、candidate logits/probs/features、GPS context、Top8 target metadata、miss label，以及 optional camera AE/image/LiDAR/radar features。
- 新增 `TopKCandidateSelector`：主方法输出 `[B, 8]` candidate scores/probs 和 `[B, 1]` miss logit，并以 `final_score_i = log p_gps(candidate_i) + lambda * modality_score_i` 保留 GPS prior。
- 新增 `CandidateAttentionSelector` ablation：candidate tokens attend 到 GPS token 与 camera AE/image tokens，作为可运行的 attention 版候选重排器。
- 新增 Top8 selector loss：candidate circular soft CE、target-in-Top8 index CE、miss BCE、GPS prior anchor KL、entropy regularization 和 hard-rank sample weighting。
- 新增训练协议 `target_adapt_beambench_top8_selector`：source scenes 可用于 pretrain，target support 用于 fine-tune，target query 只用于最终评价，GPS v2 adapter/logits frozen，camera AE/image encoder 默认 frozen。
- 新增固定 ablation matrix：至少包含 `gps_top1_baseline`、`gps_top8_oracle`、`gps_candidate_prob`、`gps_context_only_selector`、`camera_ae_gps_selector`、`camera_ae_gps_selector_anchor` 和 `top8_selector_no_gps_prior_fusion`；camera AE 不可用时跳过 camera 相关 ablation 并记录原因。
- 新增 Top8 selector summary、predictions、selection events、rank distribution、figures 和 GPS v2 comparison report，所有主结果必须与 GPS v2 r15 baseline 对比。
- 更新 README，新增 “DeepSense6G GPS Top8 Candidate Selector” 章节，说明从 residual correction 转向 Top8 selector 的原因、输入输出、loss、运行流程和结果判读方式。
- 不新增绕过 `src/kd_sensing` 包结构的旧入口或兼容聚合层；用户请求中列出的 `python -m src.*` 入口在实现时应落为包内 CLI 或薄别名，遵守现有项目架构边界。

## Capabilities

### New Capabilities

- `deepsense6g-gps-top8-candidate-selector`: 定义 DeepSense6G GPS v2 Top8 candidate manifest、candidate dataset、候选选择模型、attention ablation、loss、训练协议、固定 ablation、输出产物、可视化、GPS v2 对比报告和 query leakage guard。

### Modified Capabilities

- `project-architecture`: 新 workflow 的实现与公开入口必须位于 `src/kd_sensing/` 包内，不能新增长期维护的顶层 `src.*` 运行模块或旧式兼容聚合层。
- `modality-aware-data-loading`: 扩展 DeepSense6G optional modality path/feature 发现与缺失处理，保证 camera/LiDAR/radar 缺失时 GPS context-only Top8 selector 仍可运行。
- `experiment-workflow`: 增加 Top8 selector 的配置驱动运行、target support/query 边界、query label 防泄漏、输出 summary/report 和验收命令约束。

## Impact

- 源码：新增或修改 `src/kd_sensing/data/`、`src/kd_sensing/models/`、`src/kd_sensing/losses/`、`src/kd_sensing/engine/`、`src/kd_sensing/cli/`、`src/kd_sensing/evaluation/` 和 `src/kd_sensing/utils/` 中的 Top8 manifest、dataset、selector、loss、训练、绘图和对比逻辑。
- 配置：新增 `configs/deepsense6g_top8_selector.yaml`，默认指向 `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep` 与 `outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled/`。
- 测试：新增 Top8 manifest、selector、attention selector、loss、sparse 64 logits、circular distance 和 synthetic toy selector 测试，并复用现有 circular metrics 回归。
- 文档：更新 README，明确主方法是 GPS Top8 内候选选择，64 类 direct modality prediction 与 no-GPS-prior fusion 仅作为反例/ablation。
- 运行产物：新增 `outputs/analysis/deepsense6g_top8_selector/` 下的 manifest、summary、predictions、selection events、figures 和 comparison report；本地 outputs/logs/checkpoint/cache 仍不得纳入源码变更。
