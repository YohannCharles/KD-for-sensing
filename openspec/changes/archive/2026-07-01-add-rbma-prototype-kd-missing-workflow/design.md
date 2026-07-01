## Context

当前仓库已有 U-MaskBeamJEPA whole-model exception、AMBER full local reproduction、missing mask helper、reliability-gated cross-attention fusion、Gaussian JEPA loss、soft beam label 和 teacher guidance 非退役命名边界。用户给出的五阶段提示词不是独立新框架，而是在这些基础上增强缺失模态鲁棒 beam prediction：先解决 no-JEPA full modality 表现好但 partial modality 掉点的问题，再用 prototype alignment 和 pattern-balanced schedule 提升 `missing_gps`、`non_gps_only`、`only_gps` 等场景。

仓库现有约束：

- 项目内部 canonical 模态名称使用 `image`、`radar`、`lidar`、`gps`；不得新增 `vision` 伪模态或旧别名。
- 训练入口继续使用 `kd-sensing-train --config ...` 和现有 `TrainingExtension`；不新增根目录 `train.py`、`eval_missing_patterns.py` 或第二套训练循环。
- legacy radar KD、`logits_kd`、`rkd`、teacher/student no-KD runtime 已退役；本 change 中的 full-to-partial KD 只能作为 U-MaskBeamJEPA 内部 opt-in teacher stabilization，不能绕开 retired guard。
- 普通 baseline 不应被新增 reliability、prototype 或 KD fields 污染；新增字段只对显式 opt-in config 生效。

## Goals / Non-Goals

**Goals:**

- 增加 `reliability_biased_missing_attention` fusion，严格屏蔽缺失模态 key/value，同时对低可靠可用模态加入 log reliability bias。
- 增加 beam prototype alignment，利用 beam-neighborhood soft target 让 fused feature、可用单模态 feature 和可选 teacher feature 对齐到 beam topology。
- 增加 online full-to-partial teacher stabilization，优先服务 no-JEPA：full mask teacher stop-gradient，sampled missing mask student 正常反传。
- 增加 pattern-balanced missing sampler 和按 pattern 聚合 metrics，重点覆盖 `missing_gps`、`non_gps_only`、`only_gps`、`random_0.5`、`random_0.75`。
- 提供最小但完整的 ablation config 矩阵，能用现有训练/评估入口运行 smoke 和本地实验。
- 用 synthetic/focused tests 覆盖 shape、mask、NaN、loss backward、detach、top-k 和配置加载。

**Non-Goals:**

- 不承诺论文 AMBER 官方数值、排行榜 claim 或真实数据复现结果。
- 不实现 checkpoint teacher 的完整加载/权重管理；首轮只保留配置解析和清晰未实现错误或 disabled 状态。
- 不新增外部依赖、不复制通用 trainer、不恢复 legacy distillation runtime。
- 不要求所有模型都消费 reliability/prototype/KD metadata。
- 不把 `non_gps_only` 解释为新模态组合；在四模态设定下它与 `missing_gps` mask 相同，但日志 pattern name 必须保留区分。

## Decisions

1. RBMA attention 作为 U-MaskBeamJEPA 的窄 fusion helper，而不是新 whole-model。
   - 实现位置优先为 `src/kd_sensing/models/reliability_biased_missing_attention.py` 或现有 U-MaskBeamJEPA owner 附近的窄模块，并由 `fusion_type: reliability_biased_missing_attention` 选择。
   - 使用手写 multi-head attention：learnable beam query，keys/values 为 canonical 模态 token 加可选 global token，logits 加 `beta_reliability * log(reliability + eps)` 和 learnable modality bias。
   - hard mask 使用 boolean availability mask，缺失模态 score 置为 `torch.finfo(dtype).min` 或等价安全值，softmax 后缺失模态 attention 必须为 0。
   - 备选：扩展现有 `ReliabilityGatedCrossAttentionFusion` 的单头实现。暂不采用，因为用户明确要求 multi-head 手写实现和 attention diagnostics；但复用其 reliability bias 语义。

2. 内部模态名统一为 `image`。
   - 配置和测试使用 `["image", "radar", "lidar", "gps"]`。
   - 文档可说明附件中的 `vision` 对应项目内 `image`，但实现不得注册 `vision`。
   - 备选：在 config loader 中接受 `vision` alias。不采用，当前 U-MaskBeamJEPA spec 明确禁止伪模态。

3. Beam prototype alignment 落在 loss/helper 层，prototype bank 作为可训练模块挂在模型或 training extension state 中。
   - `BeamPrototypeBank` 保存 `[num_beams, d_model]` prototypes，使用 normalized cosine / temperature 产生 logits。
   - `make_soft_beam_labels` 复用 soft beam label 的 Gaussian/circular beam topology 语义，作为 supervised beam-neighborhood target，不记录为 KD。
   - prototype loss 输出 `loss_proto_fused`、`loss_proto_modality`、`loss_supcon`、`prototype_top1`、`prototype_top5` 和 sample counts。
   - 备选：把 prototype loss 写进 model forward。不采用，loss 权重和 ablation 属于训练配置。

4. Full-to-partial KD 首轮只实现 online full teacher。
   - 同一 batch 内先构造 `full_mask`，用 `torch.no_grad()` 或 detach 的 teacher branch 得到 `teacher_logits` 与 `teacher_feature`；再用 sampled mask 得到 student output。
   - 默认不临时切换 `model.eval()`，避免 trainer 状态复杂化；若 dropout/BN 稳定性成为实测问题，再增加配置化 teacher eval mode。
   - loss 使用 KL logit KD、feature cosine/MSE KD 和可选 prototype KD；日志命名为 `loss/full_to_partial_kd`、`loss/feature_kd`、`loss/prototype_kd`，并在 metadata 中标记为 current opt-in stabilization。
   - checkpoint teacher 仅预留配置字段 `kd_teacher_mode: checkpoint` 和 `teacher_checkpoint`，首轮遇到启用时必须给出清晰错误或 pending 状态。

5. Pattern-balanced mask sampler 复用现有 missing mask helper 边界。
   - 新增 `sample_pattern_balanced_mask(batch_size, modalities, pattern_probs, device=None, ensure_at_least_one=True)`，返回 mask、pattern_names 和可选 pattern_ids。
   - pattern 概率归一化后按样本采样；随机 pattern 必须保证至少一个模态可用。
   - trainer 只把 mask 传给模型，不原地改 batch；eval 使用显式 pattern list 构造 deterministic masks。
   - 备选：在 dataset 中物理删除模态字段。不采用，会污染 batch contract 并影响普通 baseline。

6. Ablation 配置走现有配置体系。
   - 首轮推荐四个配置：`amber_style_mask_baseline`、`no_jepa_rbma`、`no_jepa_rbma_proto`、`no_jepa_rbma_proto_kd`。
   - 小权重 JEPA 配置仅作为后续对照，不作为主实验 claim 前置条件。
   - 若当前仓库已有 `configs/fusion/` current/local baseline 入口，优先放在 `configs/fusion/experiments/...` 或当前 inventory 认可的位置；仅当项目已有 `configs/ablations/` 规范时才使用该目录。

## Risks / Trade-offs

- [Risk] `kd` 命名可能与 retired KD guard 冲突 -> metadata、logs 和 config guard 必须明确这是 U-MaskBeamJEPA opt-in full-to-partial stabilization；不得接受 `logits_kd`、`rkd` 或旧 teacher checkpoint 默认路径。
- [Risk] all-missing 样本导致 attention 全 `-inf` 和 NaN -> RBMA 必须在无 global token 时早失败，有 global token 时允许传递，测试覆盖。
- [Risk] reliability 出现 0、负数或非有限 -> attention 中 clamp/log 前做数值保护，diagnostics 记录有限性；缺失模态 reliability 仍为 0 并由 hard mask 处理。
- [Risk] supervised contrastive batch 中没有正样本导致 NaN -> 无正样本的 anchor 必须跳过，并记录有效样本数。
- [Risk] prototype top5 在 beam 数小于 5 时越界 -> top-k 使用 `min(5, num_beams)`。
- [Risk] `missing_gps` 与 `non_gps_only` mask 相同导致评估重复 -> 保留不同 pattern name 和报告行，便于对齐用户实验问题，但实现不制造额外数据语义。
- [Risk] 新增配置目录与 inventory 生命周期冲突 -> 实施时先查当前 config lifecycle，若 `configs/ablations/` 不属于 current 入口，则放入 current/local experiment 目录并在文档中说明映射。
- [Risk] 现有 active changes 未归档且工作区脏 -> 实施时只在新 change 范围内小步修改，避免覆盖用户已有 AMBER/U-MaskBeamJEPA 改动。

## Migration Plan

1. 先实现 RBMA attention 和 focused tests，确保现有 `concat_mlp`、`weighted_sum`、`reliability_gated_cross_attention` 不变。
2. 再实现 prototype alignment helper 和 loss extension，默认关闭。
3. 再实现 online full-to-partial teacher path，默认关闭；checkpoint teacher 保持 pending。
4. 最后接入 pattern-balanced sampler、eval pattern aggregation 和 ablation configs。
5. 回滚时删除新增 config 并关闭三个 feature flags，现有 U-MaskBeamJEPA 和 AMBER full 配置应继续可运行。

## Open Questions

- `BeamPrototypeBank` 应挂在 U-MaskBeamJEPA 模型内部还是 training extension state 中；实现前以最少改动为准，优先选择能随 checkpoint 保存并被 optimizer 更新的位置。
- 现有 eval matrix 是否已经能表达全部 pattern；若能复用，只扩展 pattern parser 和报告字段，不新增 CLI。
- JEPA 模式下 global token 优先使用 `mu_B` 还是 detached teacher/context token；默认使用 `mu_B`，但 no-JEPA 主路径允许无 global token。
