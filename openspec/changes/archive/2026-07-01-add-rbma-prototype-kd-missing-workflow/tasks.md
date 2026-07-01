## 1. 基线确认与接入点

- [x] 1.1 读取当前 `u_mask_beam_jepa`、AMBER full、missing mask helper、training extension 和 eval matrix owner，确认已有未提交改动，避免覆盖用户工作。
- [x] 1.2 确认 canonical 模态顺序为 `["image", "radar", "lidar", "gps"]`，并在新增配置和测试中拒绝 `vision` 伪模态。
- [x] 1.3 确认新增增强使用现有 `kd-sensing-train --config`、training extension 和 eval matrix 边界，不新增根目录训练/评估脚本。

## 2. RBMA Attention

- [x] 2.1 新增窄模块实现 `ReliabilityBiasedMissingAwareAttention`，支持 multi-head beam query attention、hard missing mask、soft reliability log-bias、learnable modality bias、dropout 和 debug diagnostics。
- [x] 2.2 在 U-MaskBeamJEPA 中新增 `fusion_type: reliability_biased_missing_attention` 分支，并保持 `concat_mlp`、`weighted_sum`、`reliability_gated_cross_attention` 行为不变。
- [x] 2.3 接入可选 global token：JEPA 模式优先使用 `mu_B`，no-JEPA 模式允许无 global token；all-missing 且无 global token 时抛清晰错误。
- [x] 2.4 新增 focused tests 覆盖 full mask、missing_gps、only_gps、non_gps_only、all_missing_with_global_token、missing attention weight 为 0、reliability log 数值有限。

## 3. Beam Prototype Alignment

- [x] 3.1 实现 `BeamPrototypeBank`，使用 normalized feature/prototype 和 temperature 输出 `[B, num_beams]` prototype logits。
- [x] 3.2 实现 `make_soft_beam_labels(labels, num_beams, sigma, circular)`，复用 beam topology 语义，保证 target 归一化并支持 circular distance。
- [x] 3.3 实现 prototype alignment loss：fused feature KL、可用模态 feature KL、可选 teacher feature loss、prototype top1/top5 和 sample count diagnostics。
- [x] 3.4 实现可选 supervised contrastive loss，跳过无正样本 anchor 并保持 loss/backward 有限。
- [x] 3.5 在 U-MaskBeamJEPA training extension 中接入 `training.use_beam_prototype_alignment`、`lambda_proto`、`lambda_modality_proto`、`lambda_supcon`、`beam_proto_temperature` 和 `beam_label_sigma`。
- [x] 3.6 新增 focused tests 覆盖 prototype forward/backward、soft target 归一化、circular wrap-around、mask 过滤缺失模态、top5 在 `num_beams < 5` 时安全。

## 4. Online Full-to-Partial Teacher Stabilization

- [x] 4.1 在 training extension 中实现 online full teacher：构造 full mask，先得到 detached teacher logits/features，再用 sampled missing mask 得到 student logits/features。
- [x] 4.2 实现 logit KD、feature KD 和可选 prototype KD，使用 `loss/full_to_partial_kd`、`loss/feature_kd`、`loss/prototype_kd` 等 current opt-in 命名。
- [x] 4.3 保证 `use_jepa_loss=false` 时不访问 JEPA-only NLL 字段，no-JEPA + KD 的 total loss backward 可运行。
- [x] 4.4 为 `kd_teacher_mode=checkpoint` 增加清晰 pending 错误或完整冻结 teacher 加载语义；不得静默恢复 legacy distillation runtime。
- [x] 4.5 新增 focused tests 覆盖 teacher detach、student 有梯度、teacher_top1/student_top1/kd_gap diagnostics 和 checkpoint teacher pending guard。

## 5. Pattern-Balanced Missing Sampler 与评估

- [x] 5.1 在现有 missing mask helper 边界中实现 `sample_pattern_balanced_mask`，返回 mask、pattern_names 和可选 pattern_ids。
- [x] 5.2 支持 `full`、`missing_gps`、`non_gps_only`、`only_gps`、`missing_one_random`、`only_one_random`、`random_0.25`、`random_0.5`、`random_0.75` 和 custom available indices。
- [x] 5.3 在训练配置接入 `training.mask_sampler: pattern_balanced` 与 `training.pattern_probs`，只传 mask 给模型，不原地修改 batch。
- [x] 5.4 扩展当前 eval matrix 或包内 eval CLI，支持显式 eval patterns 并按 pattern 汇总 top1、top5、loss 和样本数。
- [x] 5.5 新增 sampler tests：采样 1000 个样本检查比例大致符合配置、每个样本至少一个模态可用、`missing_gps` 与 `non_gps_only` mask 相同但日志 name 分离。

## 6. Ablation Configs

- [x] 6.1 按当前 config lifecycle 选择合适目录，新增 AMBER-style hard mask baseline、`no_jepa_rbma`、`no_jepa_rbma_proto`、`no_jepa_rbma_kd`、`no_jepa_rbma_proto_kd`、`jepa_small_lambda_rbma_proto_kd` 和 `proto_only_baseline` 配置或等价 current/local experiment configs。
- [x] 6.2 确保所有配置使用 canonical `image/radar/lidar/gps` 模态、当前 `model.primary` 结构、现有输出边界和 pattern-balanced sampler。
- [x] 6.3 为配置加载添加 focused tests，覆盖主候选配置、AMBER-style baseline、no-JEPA KD/prototype flags 和 retired KD override guard。

## 7. 文档与 Metadata

- [x] 7.1 更新 `training_strategy_metadata()` 或 run metadata，记录 fusion type、mask sampler、prototype alignment、full-to-partial KD、teacher mode、JEPA loss 状态和 reliability consumption。
- [x] 7.2 更新 `docs/experiment_matrix.md`，记录首轮推荐四配置和后续 JEPA 小权重对照。
- [x] 7.3 更新 `docs/experiment_protocols.md`，记录 pattern definitions、pattern probabilities、hard-label metric 口径和 canonical `image` 命名。
- [x] 7.4 更新 `docs/mainline_model_catalog.md` 和 `docs/result_claims_registry.md`，将 RBMA workflow 标记为 local/pending，避免升级为 AMBER official claim。
- [x] 7.5 如新增 config 目录、模型 owner 或 lifecycle 条目，更新 `docs/project_surface_inventory.md`。

## 8. 验证

- [x] 8.1 运行 `openspec validate add-rbma-prototype-kd-missing-workflow --strict`。
- [x] 8.2 运行新增 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py tests/test_u_mask_beam_jepa_eval_matrix.py -q` 或实施时新增的等价测试文件。
- [x] 8.3 运行配置/CLI/架构边界检查：`conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。
- [x] 8.4 对主候选配置运行最小 smoke 或 dry-run；若无法访问真实数据，只运行 synthetic/config smoke 并在最终说明记录限制。
- [x] 8.5 确认 `git status --short` 中没有新增 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 TensorBoard 产物。
