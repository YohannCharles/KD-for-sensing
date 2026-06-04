## Context

当前仓库已经完成 DeepSense6G GPS v2 adapter 与 TopK 分析，并已有 residual/camera residual 相关实现可复用：`kd_sensing.data.deepsense6g_residual` 提供 ratio tag、GPS sweep artifact 发现与 residual manifest 基础工具，`kd_sensing.data.deepsense6g_camera_residual` 提供 camera/image path 与 AE feature 关联模式，`kd_sensing.models.beam_candidate_attention` 已有候选 beam attention reranker 的雏形，`kd_sensing.losses.camera_residual_losses` 提供 support/query 防泄漏和 anchor loss 的实现参考。

项目当前架构要求新实现放在 `src/kd_sensing/` 包内，并通过包内 CLI 或 console script 暴露入口。用户给出的 `python -m src.*` 文件名应被理解为功能清单和命令语义，落地时不应新增长期维护的顶层 `src/data`、`src/models` 或 `src/run_*.py` 入口。

GPS v2 r15 的 Top8 recall 已经约为 0.8733，scenario31/33 的 Top8 recall 高，scenario32/34 是主要瓶颈。strict Top8 selector 的成功标准是：在不破坏 GPS prior 的前提下，对 target-in-Top8 样本学习更好的候选选择；对于 Top8 miss 样本，只能选择最近候选并用 miss head 诊断，不在本阶段默认扩展到 Top16。

## Goals / Non-Goals

**Goals:**

- 以 GPS v2 logits 重新计算 Top8 candidates，并生成可审计 manifest、metadata 和 Top8 recall 对齐检查。
- 提供 Top8 candidate dataset，稳定返回 candidate features、GPS context、Top8 label metadata 和 optional modality features。
- 提供 MLP selector 与 attention selector，两者都只输出 Top8 内 candidate scores/probs，不以 64 类 direct classifier 作为主方法。
- 通过 `log p_gps + lambda * modality_score` 保留 GPS prior，并用 anchor loss 限制 already-good GPS 样本被过度重排。
- 支持 source pretrain + target support finetune 的默认协议，同时保证 target query label 只用于最终评价、图表和 report。
- 输出与 GPS v2 r15 baseline 对齐的 summary、predictions、selection events、rank distribution、figures 和 comparison report。

**Non-Goals:**

- 本阶段不默认实现 Top16 fallback、miss-triggered fallback 或动态候选扩展。
- 不把 image、LiDAR、radar 作为主方法直接预测 64 类 beam。
- 不恢复蒸馏训练、旧 KD 配置或绕过 `kd_sensing` 包结构的旧入口。
- 不要求 camera AE、image token、LiDAR、radar 全部可用；缺失时必须保留 GPS context-only baseline。

## Decisions

### Decision 1: Top8 必须来自真实 GPS v2 logits

manifest builder 新增在 `kd_sensing.data.deepsense6g_topk_candidates`，复用 `ratio_tag` 与 `PREDICTION_LOGIT_NAMES` 等命名约定，但实现独立的 strict logits loader。默认 `candidate.require_saved_logits=true`，当 `gps_logits.npy`、`logits.npy`、`pred_logits.npy` 或 logits index 不可用时直接报错，并提示重跑 GPS v2 with `--save-logits`。

备选方案是沿用 residual workflow 的 fallback Gaussian prior。这里不采用，因为 Top8 candidate selection 的召回上限必须来自真实 GPS logits；用 top1 Gaussian 会把候选生成质量和 selector 质量混在一起，无法与已有 TopK analysis 对齐。

### Decision 2: Top8 selector 是独立 workflow，不继续扩张 residual 主线

新增能力命名为 `deepsense6g-gps-top8-candidate-selector`，配置默认写入 `outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled/`。它可以复用 camera residual 的 path/AE feature 发现方式，但输出表、loss、模型和 comparison report 都围绕 Top8 hit/miss 与候选 rank 设计。

备选方案是在 `deepsense6g-gps-residual-fusion` 内增加一个 ablation。这里不采用，因为新的主问题已经从 64 类 residual correction 转为候选内选择，直接塞入 residual spec 会让验收指标、训练协议和报告语义变得含混。

### Decision 3: 模型输出以 candidate score 为主，sparse 64 logits 只做兼容

`TopKCandidateSelector` 输入 candidate features、GPS context、candidate log probs 和 optional modality embeddings。输出 `final_candidate_scores [B, 8]`、`modality_candidate_scores [B, 8]`、`candidate_probs [B, 8]`、`miss_logit [B, 1]`、`lambda_value` 和 diagnostics。为了复用现有 metrics，可提供 helper 将 Top8 scores scatter 到 `[B, 64]` sparse logits，非候选 beam 填 `-1e9`。

备选方案是让模型直接输出 `[B, 64]` 并在 loss 中限制 Top8。这里不采用，因为它会弱化“候选内选择”的实验问题，也更容易让 modality 覆盖 GPS prior。

### Decision 4: 训练 split 以 source/support 为唯一可学习集合

默认 `source_pretrain_target_finetune`：source scenes 训练 selector，target scene support fine-tune，target support 内部分出 validation 做 early stopping。`support_only` 是必备快速 baseline，`source_plus_support` 可作为扩展模式。target query 不进入 normalization fit、loss、early stopping 或模型选择。

备选方案是用 target query 参与 scaler 或 validation 以稳定训练。这里不采用，因为当前 workflow 的可信度核心是 target query leakage guard。

### Decision 5: Optional modality 采用“可用即启用，不可用即记录跳过”

camera AE feature 优先作为第一版 optional modality；image tensor/tokens、LiDAR/radar feature 保留 dataset 字段和配置开关。每个 ablation 写入实际启用 modalities 与 `skipped_reason`，camera 缺失时仍运行 `gps_context_only_selector`、`gps_top1_baseline`、`gps_top8_oracle` 和 `gps_candidate_prob`。

备选方案是让 camera AE 成为硬依赖。这里不采用，因为现阶段必须先验证候选 pipeline 和 GPS context-only selector。

## Risks / Trade-offs

- [Risk] GPS v2 r15 目录缺少 logits，strict manifest 无法生成 → Mitigation: CLI 早失败并输出需要重跑 GPS v2 with `--save-logits` 的明确提示，同时保留 inspection/report 字段说明缺失 artifact。
- [Risk] scenario32/34 的 strict Top8 recall 上限过低，selector 无法提升 → Mitigation: summary 按 scene 与 Top8 hit/miss 分组，comparison report 必须判断是否需要后续 `Top8 primary + miss-triggered Top16 fallback`，但本 change 不默认实现 fallback。
- [Risk] camera AE feature 与样本/时间戳对齐不稳定 → Mitigation: manifest 记录 image path、image_exists、AE feature row index、feature fingerprint 和 skipped/fallback reason，并在 report 中比较 camera AE selector 与 GPS context-only selector。
- [Risk] selector 过度破坏 GPS already-good 样本 → Mitigation: 默认保留 GPS prior fusion、`lambda_max` clamp、good-sample anchor KL 和 good-sample degradation 指标；`top8_selector_no_gps_prior_fusion` 仅作为反例 ablation。
- [Risk] 新 CLI 与用户给出的 `python -m src.*` 名称不一致 → Mitigation: 正式入口采用 `kd_sensing.cli.*` 与 console scripts；文档中给出包内 `python -m kd_sensing.cli...` 与 `kd-sensing-*` 命令。

## Migration Plan

1. 新增配置、manifest builder、dataset、models、loss 和 CLI，不修改 GPS v2 既有结果目录。
2. 在实现早期先跑 manifest 与 baseline/oracle ablation，确认 GPS top1 baseline 能复现 r15 指标、Top8 recall 与已有 TopK analysis 对齐。
3. 接入 MLP selector 与 loss，再接入 attention selector ablation。
4. 接入 plot 与 comparison report，最后更新 README。
5. 如需回滚，删除新 `deepsense6g_top8_selector` 配置、CLI、模块和 OpenSpec change；既有 GPS v2、residual 和 camera residual workflow 不受影响。

## Open Questions

- GPS v2 sweep 当前 r15 目录是否已经保存完整 logits 与 index；如果没有，需要先补跑 GPS v2 保存 logits。
- camera AE feature 的最终目录与 row index 文件是否已经稳定；若尚未稳定，第一版默认以 GPS context-only selector 完成 strict pipeline 验收。
- source scene logits 是否覆盖所有 source rows；若 source prior 不完整，默认降级为 `support_only` 并在 metadata 记录原因。
