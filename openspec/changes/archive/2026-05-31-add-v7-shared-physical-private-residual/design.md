## Context

现有 HiST-Beam 主线集中在 `hist_beam_fusion` 模型、`hist_beam_losses`、`hist_beam_adaptation`、LOSO 执行与评估输出中。v3/v4/v6/v8 已覆盖 shared/private、adapter、radio/path prototype 和 full fine-tuning，但 shared 分支主要通过 coarse/hierarchical beam loss 学语义，不直接学习可迁移的 beamspace power distribution；private 分支在 adapter 适配中也可能承担过多 beam 分类能力。

项目已有 `beam_soft_targets` 与 `soft-beam-label-training` 规格，规定 target 域训练不得读取 target-side power/RSS oracle。v7 需要复用这个边界：BSP 物理标签主要服务 source pretraining；target few-shot adaptation 默认只用 hard beam label、residual 正则和 gate 正则，不把 target power/RSS 当训练 oracle。

用户明确要求不要加入历史 label，因此本设计不启用 `hist_beam.history_anchor`，不把 `input_beam`、last beam 或 residual-delta 历史目标作为 v7 输入或监督。现有 history-anchor 测试和实现保持独立。

## Goals / Non-Goals

**Goals:**

- 新增 `v7_shared_physical_private_residual`，在现有 `hist_beam_fusion` 注册入口内构建，不破坏 v3/v4/v6/v8。
- 新增 `beamspace_power_label` 数据契约，优先复用 MMW/DeepSense6G 中已有 beam power vector，缺失时可通过 path/AoD 近似构造，并保存可诊断缓存。
- 让 shared 分支同时输出 `logits_shared` 和 `pred_beamspace_power`，并保证 shared-only 可独立评估。
- 让 private 分支只输出 `delta_logits_private`，通过 `alpha` gate 叠加到 shared logits 得到 `logits_final`。
- 为 source training、target adaptation、evaluation 和 LOSO summary 增加 v7 专用 loss、冻结策略与诊断字段。

**Non-Goals:**

- 不实现或启用历史 label、history anchor、last-beam residual target。
- 不让 private 分支直接输出完整 beam prediction。
- 不把 target-side beam power/RSS/path oracle 作为默认 target adaptation 训练监督。
- 不移除或重命名现有 v3/v4/v6/v8 配置和产物语义。
- 不把 Beam-Delay Power Profile 作为第一阶段必需能力；只预留配置和接口边界。

## Decisions

1. **v7 作为 `hist_beam_fusion` 的新 variant，而不是新模型注册名。**

   现有 LOSO、配置、训练扩展和评估流程已经围绕 `hist_beam_fusion` 分发 variant。继续在 `src/kd_sensing/models/fusion/hist_beam.py` 扩展 `HIST_BEAM_VARIANTS`、`HistBeamConfig` 和 `HistBeamFusionNet`，可复用 modality encoder、token transformer、adapter 和 summary 管线。备选方案是注册单独模型类，但会复制大量构建和 LOSO 执行逻辑，增加维护面。

2. **BSP 标签先复用 beam power vector，再做 path/AoD fallback。**

   `MMWDataset` 已有 `_load_beam_power`、`return_beam_power`、radio semantic 逻辑，第一优先级应把有效 beam power/RSS vector 归一化为 `beamspace_power_label`。缺失时新增窄 helper 解析 path npz/json 可用字段，并支持配置指定 key；如果缺少真实 codebook，先实现 AoD 到 beam bin 的平滑近似。备选方案是只使用 Gaussian hard-label smoothing，但这会退化成普通 soft target，不能表达物理传播功率分布。

3. **BSP 与现有 `target_beam_distribution` 分离命名。**

   `target_beam_distribution` 是 beam soft supervised target，已有 source/target 域规则。v7 的 `beamspace_power_label` 是 shared physical supervision，日志和 metadata 使用 `physical_label/*`、`hist/v7/*` 命名，避免被误认为 KD soft label 或历史 label。

4. **forward 输出以 final logits 保持兼容，同时暴露 shared/residual 诊断。**

   v7 forward 返回 `logits` 和 `beam_logits` 指向 `logits_final`，以兼容现有训练与评估入口；同时返回 `logits_shared`、`delta_logits_private`、`alpha`、`pred_beamspace_power`、`shared_representation`、`private_representation` 和 `adapter_representation`。shared-only 指标由评估层显式读取 `logits_shared` 计算。

5. **source warmup 先禁用 private residual。**

   `training.shared_warmup_epochs` 内只训练 shared hard CE、beamspace soft KL 和 physical head KL，`logits_final` 视为 `logits_shared`。warmup 后再加入 residual、gate 和 difference 正则，降低 private 一开始偷学完整分类的风险。

6. **target adaptation 使用白名单冻结策略。**

   新增策略名 `v7_private_residual`，冻结 encoders、fusion transformer、shared branch、shared beam head 和 physical head；只训练 `private_adapter`、`private_residual_head`、`residual_gate`，以及配置允许的 norm affine 参数。这样 v7 与 full fine-tuning baseline 保持可比，并能从 trainable ratio 诊断效率。

7. **target BSP 默认只做诊断，不做训练 oracle。**

   如果 target batch 中存在 `beamspace_power_label`，evaluation 可以计算 `phys_kl`；target adaptation 默认不使用它反传。只有显式 diagnostic/ablation flag 才允许记录或加入冻结 shared 下的 consistency 项，且 metadata 必须标记使用边界。

8. **多 source 默认做 scene-balanced sampling，V7 source CE 默认做 class balance。**

   单一 source 或简单 concat 多 source 会把训练 prior 固定在 source 主 beam 区域。v7 stage defaults 因此启用 source scene-balanced weighted sampler，并在 source hard CE 上使用 batch-level inverse-sqrt class weights；target adaptation 默认不启用 class balance，避免 10-shot 小样本权重噪声进一步放大。

9. **V7 few-shot 默认用 beam-frequency stratification。**

   原有 sampler 在 radio semantic 可用时优先按 radio semantic 抽样。V7 的 cross-scene collapse 主要来自 source/target beam 分布错位，因此新增 `hist_beam.adaptation.few_shot_stratification`，V7 默认选择 target adapt pool 中频率最高的 beam 类各抽一个样本，以提高少样本集合覆盖 target 主 beam 的概率。

## Risks / Trade-offs

- [Risk] path/AoD fallback 与真实 codebook 不一致，可能造成 BSP top1 与 hard beam 低一致。→ Mitigation: 优先使用 beam power vector；fallback 必须写入 `beamspace_power_source` 和 entropy/top1 agreement 诊断，不把一致率作为硬门槛。
- [Risk] v7 loss 项过多导致早期训练不稳定。→ Mitigation: 提供 warmup 和保守默认权重，并在日志中分项记录 loss；测试覆盖 loss 可关闭/权重为 0 的路径。
- [Risk] `logits` 指向 final 后，现有 Top-K 默认不看 shared-only。→ Mitigation: 评估层新增 `shared_topk` 和 `final_topk`，predictions/metrics 保留 final 主指标。
- [Risk] target power/RSS 泄漏到 adaptation。→ Mitigation: 复用现有 sensitive field policy，新增 v7 防泄漏测试；target adaptation 默认不读取 `beamspace_power_label` 作为训练 loss。
- [Risk] v7 与 history-anchor 残差字段命名混淆。→ Mitigation: v7 使用 `private_residual_head`、`delta_logits_private`、`residual_gate`，不使用 `last_beam`、`residual_logits` 或 `history_anchor` 训练路径。
- [Risk] beam-frequency few-shot 使用 target adapt pool label 分布选择样本，可能高估随机标注流程下的可达效果。→ Mitigation: 该策略通过显式 `few_shot_stratification` 配置和 sampling manifest 记录，实验结论需与 radio/random/coarse 策略分开报告。

## Migration Plan

1. 新增 v7 配置、模型字段和 BSP 数据 helper，默认关闭，不影响现有配置。
2. 扩展训练 loss 和 adaptation strategy，仅当 variant 或 adaptation method 为 v7 时启用。
3. 扩展 evaluation/LOSO summary，缺少 v7 字段时保持现有行为。
4. 用 focused tests 验证 v7 构建、forward shape、loss、冻结白名单、BSP 生成和 target oracle 防泄漏。
5. 回滚时删除 v7 配置和 variant 分支即可；现有 v3/v4/v6/v8 路径不依赖 v7。

## Open Questions

- MMW path 文件中 AoD/complex gain 字段命名是否在所有场景稳定；第一版需要打印可用 keys 并支持 config override。
- 是否需要在后续阶段实现 beam-wise gate `[B, H, C]`；第一版采用 scalar gate `[B, H, 1]`。
- Beam-Delay Power Profile 是否进入下一轮 change；当前仅保留配置 flag 和非主实验接口。
