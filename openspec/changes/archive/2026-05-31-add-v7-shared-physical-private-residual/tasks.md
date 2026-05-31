## 1. 数据与物理标签

- [x] 1.1 新增 beamspace physical label 配置解析，支持 `enabled`、`required`、`eps`、`temperature`、`smoothing_sigma`、`source`、`cache_dir`、path 字段映射和 dB/linear power 单位。
- [x] 1.2 新增 BSP 构造 helper，优先将 beam power/RSS vector 归一化为 `beamspace_power_label`，并覆盖非有限、全零、维度错误和 dB 转线性测试。
- [x] 1.3 新增 path/AoD fallback helper，支持读取 path payload keys、config field_map、AoD bin 量化、Gaussian smoothing 和不可用原因。
- [x] 1.4 在 MMW dataset 中接入 `physical_label`，返回 `beamspace_power_label`、availability mask、source 和 unavailable reason，同时保持现有 `beam_power`、radio/path semantic 行为兼容。
- [x] 1.5 实现 scene-level physical label cache 读写和 metadata 校验，缓存路径遵循 `cache/physical_labels/<dataset_name>/<scene_name>/beamspace_power_<num_classes>.npz`。
- [x] 1.6 增加首次构造统计日志或 metadata：样本数、beam 数、entropy mean/std、BSP top1 与 hard beam agreement。

## 2. V7 模型

- [x] 2.1 扩展 `HIST_BEAM_VARIANTS`、`HistBeamConfig` 和配置解析，注册 `v7_shared_physical_private_residual`，并默认禁用 `history_anchor`。
- [x] 2.2 在 `HistBeamFusionNet` 中新增 shared beam head、physical beamspace head、private residual head、scalar residual gate 和 residual scale 配置。
- [x] 2.3 实现 V7 forward：输出 `logits_shared`、`delta_logits_private`、`alpha`、`logits_final`、`pred_beamspace_power`，并让 `logits`/`beam_logits` 指向 final logits。
- [x] 2.4 确保 private residual 不作为完整 beam classifier 输出，`delta_logits_private` 只通过 gate 与 shared logits 合成 final logits。
- [x] 2.5 增加模型构建和 forward shape 测试，覆盖 image/gps/lidar 与 image/radar/gps 模态组合。

## 3. Loss 与训练流程

- [x] 3.1 新增或扩展 `compute_hist_beam_loss` 的 V7 分支，计算 shared CE、final CE、shared BSP KL、physical head KL、residual L2、gate L1 和 shared/private difference loss。
- [x] 3.2 在训练扩展中准备 `beamspace_power_label` 和 mask，确保 V7 source training 消费 BSP，普通 HiST-Beam 和非 HiST 配置不受影响。
- [x] 3.3 实现 `training.shared_warmup_epochs` 行为，warmup 内禁用 private residual loss，并让 final 等价 shared。
- [x] 3.4 为 V7 loss 增加 diagnostics 字段，包括 `hist/v7/loss_shared_ce`、`hist/v7/loss_final_ce`、`hist/v7/loss_bsp_kl`、`hist/v7/loss_phys_kl`、`hist/v7/loss_res_l2`、`hist/v7/loss_gate_l1` 和 `hist/v7/loss_diff`。
- [x] 3.5 增加 loss 单元测试，覆盖 BSP 有效、BSP 缺失、warmup、权重为 0 和 batch 太小时 difference loss 关闭。

## 4. Target Adaptation

- [x] 4.1 新增 adaptation strategy `v7_private_residual`，冻结 encoders、fusion transformer、shared branch、shared beam head 和 physical head。
- [x] 4.2 设置 V7 默认 trainable 白名单为 `private_adapter`、`private_residual_head`、`residual_gate` 和配置允许的 norm affine 参数。
- [x] 4.3 实现 V7 target adaptation loss：final hard CE、residual L2、gate L1；默认不使用 target-side `beamspace_power_label` 反传。
- [x] 4.4 扩展 leakage diagnostics，记录 `uses_input_beam_as_model_input=false`、`used_target_physical_label_for_training=false` 和 target physical oracle 未使用原因。
- [x] 4.5 增加冻结白名单和 target oracle 防泄漏测试。

## 5. Evaluation、Predictions 与 Summary

- [x] 5.1 扩展 evaluation 读取 V7 diagnostics，分别计算 `shared_top1`、`shared_top3`、`final_top1`、`final_top3`。
- [x] 5.2 若 beam power vector 可用，分别计算 shared/final NRP 和 beam power loss dB；不可用时记录原因而不伪造指标。
- [x] 5.3 计算并写出 `alpha_mean`、`alpha_std`、`delta_norm` 和可用时的 `phys_kl`。
- [x] 5.4 扩展 prediction writer，V7 predictions 包含 final/shared predicted beam、final/shared top-k、sample id、scene、split 和 variant metadata。
- [x] 5.5 扩展 LOSO summary/conclusion，使 v7 可与 v3/v4/v6/v8/full-finetune baseline 横向比较。

## 6. 配置、文档与验证

- [x] 6.1 新增 `configs/hist_beam/v7_shared_physical_private_residual.yaml`，包含 model、physical_label、loss、training warmup 和 adaptation 默认配置。
- [x] 6.2 将 MMW sensor-assisted quick validation 的可选 variants 示例扩展为包含 v7，但不改变现有默认矩阵，避免破坏已有实验入口。
- [x] 6.3 增加或更新 focused tests：BSP helper、MMW dataset BSP contract、V7 model forward、V7 loss、V7 adaptation freeze、V7 evaluation metrics。
- [x] 6.4 运行 `openspec validate add-v7-shared-physical-private-residual --strict` 并修复规格问题。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest tests/test_hist_beam_loso.py tests/test_history_anchored_residual_beam.py tests/test_p3_path_semantics.py -q`，确认现有 v3/v4/v6/v8 和 history-anchor 路径未被破坏。
- [x] 6.6 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_component_registry.py -q`，确认注册、导入边界和架构约束仍通过。
- [x] 6.7 根据变更风险运行 `conda run -n kd_mm_beam pytest -q` 或记录未运行原因。

## 7. 多 source 与 collapse 缓解

- [x] 7.1 为 LOSO source train loader 增加可配置多 source scene-balanced weighted sampler，避免简单 concat 被最大 source scene 主导。
- [x] 7.2 为 V7 source training 增加可配置 class-balanced hard CE，默认 source 开启、target adaptation 关闭，并写出 class-balance diagnostics。
- [x] 7.3 为 target few-shot sampler 增加 `beam_frequency` stratification，可通过 `hist_beam.adaptation.few_shot_stratification` 覆盖默认 radio semantic 优先策略。
- [x] 7.4 将 V7 stage defaults 设置为 beam-frequency few-shot、source scene balance 和 source class balance，并更新 V7 独立配置示例。
- [x] 7.5 增加 focused tests 覆盖 beam-frequency few-shot、多 source scene-balanced sampler 和 V7 class-balance loss diagnostics。
