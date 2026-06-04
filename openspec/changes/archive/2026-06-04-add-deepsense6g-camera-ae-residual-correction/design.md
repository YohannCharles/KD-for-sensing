## Context

DeepSense6G scenario31-34 已有 GPS v2 target-adapt support sweep，r15 的 overall 结果约为 `DBA=0.6707`、`mean err=2.344`、`P(error<4)=0.8523`，说明 GPS v2 是强 coarse anchor。当前仓库也已有 `deepsense6g_residual_fusion` 的 GPS-anchored residual workflow；本 change 是下一阶段 camera-assisted 增量，不应重写 GPS v2 adapter，也不应把 camera 从零 64 类分类作为主路线。

仓库架构要求所有运行实现位于 `src/kd_sensing/` 包结构内。因此需求文本中的 `python -m src.*` 命令在实现时改为 `python -m kd_sensing.cli.*` 或 console script，功能、参数和输出语义保持等价。

## Goals / Non-Goals

**Goals:**

- 构建 `configs/deepsense6g_camera_residual.yaml`，默认使用 scenario31-34、`mapping_disabled`、`support_ratio=0.15`、64 beam circular label 和 GPS v2 r15 prior。
- 生成 camera residual manifest，合并 GPS v2 predictions/logits/fallback prior、support/query split、GPS context、target label、image path 和 AE feature index。
- 训练 tiny Camera AE，导出 frozen AE feature，并在 residual/gate 阶段冻结 GPS v2 prior 与 AE encoder。
- 让 camera 只预测 local residual delta、correction gate 或 candidate rerank score；默认重点优化 `gps_error >= 4` 的 hard samples，并用 good-anchor 保护 `gps_error < 4` 的 samples。
- 提供可运行 ablation、summary、predictions、correction events、candidate recall、figures 和 GPS v2 comparison report。
- 明确 query label 只用于最终 evaluation 和诊断图，不用于 prior 构造、split 生成、early stopping、模型选择或调参。

**Non-Goals:**

- 不重写 GPS v2 adapter，不改变既有 GPS v2 r15/r20 sweep 口径。
- 不把 image-only direct beam classifier 作为推荐主方法；它只保留为反例 ablation。
- 不下载 pretrained weights，不依赖互联网，不提交 dataset、outputs、logs、checkpoint、cache 或 AE feature 产物。
- 不新增顶层 `src.*` 运行入口或绕过 `kd_sensing` 包结构的兼容层。
- 不要求 Stage C reranker 必须超过 Stage B；第一版只要求最小可运行和候选召回可解释。

## Decisions

### Decision 1: 在现有 GPS residual workflow 上做 camera 增量

实现复用已有 GPS v2 artifact loader、circular metric、fallback Gaussian prior、summary comparison 和 residual manifest 经验，但新增独立 camera residual config/output root，避免把历史 `deepsense6g_residual_fusion` 结果口径改乱。

替代方案是直接扩展已有 `configs/deepsense6g_residual_fusion.yaml`。该方案会把 optional image encoder 与 AE feature stage 混在同一个 workflow 中，不利于区分 “GPS context residual baseline” 与 “camera AE residual stage”，因此不作为默认。

### Decision 2: manifest 是 camera stage 的唯一数据边界

新增 manifest builder 输出：

```text
outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/manifest/camera_residual_manifest.csv
outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/manifest/camera_residual_manifest_with_ae.csv
```

manifest 每行一个样本，记录 GPS prior source、GPS top-K、GPS residual/good-bad、image availability、AE feature path/index 和 split role。训练 Dataset 只消费 manifest，不回头扫描 GPS v2 输出目录，以便审计 leakage 与复现实验。

### Decision 3: Camera AE 是无监督 feature stage，默认非 transductive

AE 训练默认使用 source scenes 的 image 和 target support image；`ae.use_target_query_unlabeled=false`。如用户显式开启 target query unlabeled image，metadata 必须记录 transductive 设置，但仍不得使用 query label。

AE 使用 tiny convolutional autoencoder，不依赖 torchvision pretrained weights。输出 checkpoint、metrics 和 reconstruction examples，feature extraction 再生成 `features.npy` 与 `features_index.csv`。

### Decision 4: Residual/gate 使用 local delta 分布而不是 64 类 correction logits

主模型 `CameraGPSResidualFusion` 输出 `residual_delta_logits: [B, 2R+2]`、`correction_gate: [B, 1]`、`p_corr: [B, 64]` 和 `final_logits: [B, 64]`。默认 `R=8`，类别为 `[-8, +8]` 加 overflow。`p_corr` 由 `gps_pred_top1 + delta mod 64` 合成；overflow 默认均匀分配到 64 beams 或按配置忽略。

替代方案是继续使用已有 64 类 `correction_logits` 加到 GPS logits。该方案表达能力更强，但较容易在少量 support 下覆盖 GPS good samples。本阶段优先使用 local delta distribution，让 correction 行为更可解释。

### Decision 5: Gate 初始更相信 GPS，good-anchor 保护 already-good

`gate_head` bias 默认初始化为 `-2.0`，让模型训练初期偏向 GPS prior。loss 组合包含：

```text
L = final_ce + residual_ce + gate_bce + good_anchor_kl + optional_aux + optional_gate_entropy
```

final CE 使用 circular soft CE 并对 GPS hard samples 加权；gate target 只在训练/support 样本上由 `gps_error >= 4` 生成；good-anchor KL 只作用于 GPS good samples，约束 final distribution 不要偏离 GPS prior。

### Decision 6: Stage C reranker 只在 GPS 候选集合内重排

`BeamCandidateAttentionReranker` 的候选集合为 GPS top-K 与 GPS top1 local circular window 的 union。第一版 image representation 可用 AE feature 复制为 pseudo token，后续可兼容 patch tokens。reranker 只在 candidate set 内打分，并必须报告 `target_in_gps_top16`、`target_in_local_radius8` 和 `target_in_union_candidates`，避免把候选召回不足误判为模型失败。

### Decision 7: 包内 CLI 分阶段运行

推荐入口为：

```bash
conda run -n kd_mm_beam python -m kd_sensing.cli.prepare_deepsense6g_camera_residual_manifest --config configs/deepsense6g_camera_residual.yaml --support-ratio 0.15 --label-space mapping_disabled
conda run -n kd_mm_beam python -m kd_sensing.cli.train_deepsense6g_camera_ae --config configs/deepsense6g_camera_residual.yaml --support-ratio 0.15 --label-space mapping_disabled
conda run -n kd_mm_beam python -m kd_sensing.cli.extract_deepsense6g_camera_ae_features --config configs/deepsense6g_camera_residual.yaml --support-ratio 0.15 --label-space mapping_disabled
conda run -n kd_mm_beam python -m kd_sensing.cli.run_deepsense6g_camera_residual --config configs/deepsense6g_camera_residual.yaml --support-ratio 0.15 --label-space mapping_disabled
```

plot 与 compare 使用同样的包内 CLI。是否额外注册 console script 由实现阶段根据 `pyproject.toml` 维护成本决定，但 console script 必须委托包内 CLI。

## Risks / Trade-offs

- [Risk] 本地 image path 在 scenario31-34 中路径结构不一致。→ Mitigation：manifest builder 支持多候选列和常见目录发现，找不到 image 时仍生成 manifest，训练 image/AE 阶段给出清晰错误或跳过缺失样本。
- [Risk] GPS v2 r15 产物未保存 logits，Gaussian fallback prior 不如真实 logits。→ Mitigation：manifest 显式记录 `gps_prior_source`，优先读取 `gps_logits.npy`、`logits.npy`、`pred_logits.npy` 和 index，缺失时只用 GPS top1 构造 prior。
- [Risk] AE 重建 feature 未必对 hard residual 有帮助。→ Mitigation：默认 ablation 包含 `gps_context_only_residual`、camera direct 反例、gated/anchor/source-pretrain 版本，summary 报告 camera 相对 GPS context 的真实增益。
- [Risk] residual/gate head 在 target support 上过拟合并破坏 good samples。→ Mitigation：gate bias、good-anchor KL、hard sample metrics、good degradation rate 和 no-gate ablation 联合诊断。
- [Risk] query leakage 难以肉眼确认。→ Mitigation：split_role 写入 manifest，loss/early stopping 只接受 train/support mask，测试覆盖 query label 不参与 gate target、residual label training 和 model selection。
- [Risk] Stage C candidate recall 太低导致 reranker 看似失败。→ Mitigation：必须输出 candidate recall，并将 reranker 作为可选阶段，不作为 Stage A/B 验收门槛。

## Migration Plan

1. 新增 OpenSpec 规格、配置和 README/实验说明，确认 camera residual scope。
2. 实现 camera residual manifest builder，先能复现 `gps_prior_only` r15 baseline。
3. 实现 Camera AE、训练 CLI、feature extraction CLI 和 manifest with AE 回写。
4. 实现 local delta class helper、`CameraGPSResidualFusion`、loss、Dataset/DataLoader 和 Stage B 训练/eval。
5. 实现 Stage C 最小 attention reranker、plots、compare report 和 ablation summary。
6. 运行单测、OpenSpec strict validate、GPS v2/circular 回归和必要 smoke 命令。

Rollback 方式是删除新增 camera residual config/CLI/model/loss/tests 和 OpenSpec change；已有 GPS v2 adapter 与 `deepsense6g_residual_fusion` workflow 不需要回滚。

## Open Questions

- 本地 DeepSense6G scenario31-34 的 image path 是否覆盖所有 GPS v2 sample；如果覆盖不足，Stage A/B 的训练集规模需要在 manifest metadata 中明确。
- AE 是否需要支持 target query unlabeled image 的 transductive ablation；默认关闭，但配置保留开关。
- overflow delta 类第一版采用均匀分配还是 ignore contribution，需要实现时通过配置固定默认并在 tests 中锁定。
- 是否注册 `kd-sensing-*` console scripts，还是只提供 `python -m kd_sensing.cli.*`；实现时可根据 README 可读性和 pyproject 维护成本决定。
