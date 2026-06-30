## 1. 模型与注册

- [x] 1.1 新增 `src/kd_sensing/models/u_mask_beam_jepa.py`，实现 `ModalityReliabilityHead`、`FullModalTeacher`、`SetContextEncoder`、`GaussianJEPAPredictor`、`ReliabilityGatedCrossAttentionFusion`、`BeamPredictionHead` 和 `UMaskBeamJEPA`。
- [x] 1.2 将 `u_mask_beam_jepa` 注册到 `MODELS`，并在 `src/kd_sensing/registries.py` 的 `import_default_components()` 中加入默认组件导入。
- [x] 1.3 实现 canonical 模态校验，默认使用 `image`、`radar`、`lidar`、`gps`，拒绝未知、重复、空模态和 `vision` 伪模态。
- [x] 1.4 实现 `training_strategy_metadata()`，记录 whole-model exception、启用模态、missing mask/reliability metadata 消费、teacher/JEPA/uncertainty/fusion/context ablation 状态。

## 2. Mask 与 Corruption Helper

- [x] 2.1 新增 `src/kd_sensing/data/missing_mask.py`，实现 `sample_missing_mask` 和 `make_pattern_mask`，覆盖 float/list p_missing、always-available indices、ensure-at-least-one 和 shape/dtype。
- [x] 2.2 实现 `apply_modality_corruption(batch, corruption_config)` 的第一版非原地拷贝逻辑，支持 image Gaussian noise/zero out、gps Gaussian noise、lidar/radar placeholder dropout。
- [x] 2.3 将训练 missing mask 注入设计接到现有 training extension 或 `ForwardControls.model_kwargs`，避免新增训练循环分支。

## 3. Loss 与训练指标

- [x] 3.1 新增 `src/kd_sensing/losses/u_mask_beam_jepa.py`，实现 beam CE、teacher CE、Gaussian latent NLL 和总损失组合，支持 `lambda_teacher`、`lambda_jepa`、`logvar_min/max`。
- [x] 3.2 将 U-MaskBeamJEPA loss 作为 opt-in training extension 或 prediction loss 扩展接入现有 `BatchStepRunner`，不恢复 `logits_kd`、`rkd` 或 distiller registry。
- [x] 3.3 记录 diagnostics：`loss_beam`、`loss_teacher`、`loss_jepa`、top1/top5 acc、mean modality reliability、mean global reliability 和 disabled ablation 状态。

## 4. 配置与 Ablation

- [x] 4.1 新增 U-MaskBeamJEPA smoke config，使用现有 `kd-sensing-train` 入口和 ignored `outputs/` 产物边界。
- [x] 4.2 新增最小 ablation overlays：`no_jepa`、`no_uncertainty`、`concat_mlp`、`weighted_sum`。
- [x] 4.3 支持 eval missing pattern 配置字段，例如 available modalities 或 explicit pattern mask，并保持普通 baseline 不要求该字段。
- [x] 4.4 对未实现的 `mask_transformer` 或未知 `fusion_type/context_type` 抛出清晰错误。

## 5. Focused Tests

- [x] 5.1 新增 synthetic registry build 和 forward 测试，验证 `adapt_model_output` 可消费输出、字段 shape 正确、全 0 mask 被拒绝。
- [x] 5.2 新增 loss backward 测试，验证随机 tensor 输入可反传，关闭 teacher/JEPA/uncertainty 后仍可训练 student logits。
- [x] 5.3 新增 missing mask helper 测试，验证 ensure-at-least-one、always-available、pattern mask 和 corruption 非原地修改。
- [x] 5.4 新增 metadata/architecture summary 或 architecture boundary 覆盖，确保 whole-model exception 可审计且未新增旧入口。

## 6. 验证

- [x] 6.1 运行 `openspec validate add-u-mask-beam-jepa --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_u_mask_beam_jepa.py -q`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_component_registry.py -q`。
- [x] 6.4 如触碰训练 extension 或 runtime，追加运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_prediction_objectives.py -q`。

## 7. 真实训练性能 Follow-up

- [x] 7.1 新增 opt-in DeepSense6G sample-level LMDB cache 预处理器，减少 image/radar/lidar/gps 多小文件读取。
- [x] 7.2 Dataset 支持 `data.dataset.sample_cache` 显式读取 LMDB，默认关闭。
- [x] 7.3 训练支持 `training.validation.interval_epochs` 降低每 epoch validation 频率，并保留最后 epoch 验证。

## 8. Modality Reliability Head 修正

- [x] 8.1 将 `ModalityReliabilityHead` 从 scalar reliability 改为输出 per-modality `mu_i_B` 与 `logvar_i_B`。
- [x] 8.2 将 reliability 定义改为 `exp(-mean(softplus(logvar_i_B), dim=-1))`，并保持缺失模态 reliability 为 0。
- [x] 8.3 将 JEPA latent NLL 改为只对 `missing_mask_i=1` 的 per-modality uncertainty loss 求平均。
- [x] 8.4 补充 focused tests 覆盖 per-modality 输出、reliability 公式和缺失模态不参与 uncertainty loss。

## 9. 完整 JEPA NLL 与 Scenario 32 配置修正

- [x] 9.1 将 U-MaskBeamJEPA loss 拆分为 `loss_jepa_global` 和 `loss_modality_nll`，总损失使用 `lambda_jepa_global` 与 `lambda_modality_nll` 加权。
- [x] 9.2 将 `modality_reliability` 输出形状修正为 `[B, M, 1]`，并保持缺失模态 reliability 为 0。
- [x] 9.3 新增 `configs/fusion/u_mask_beam_jepa_s32.yaml`，保留 scene 31 smoke config 不覆盖。
- [x] 9.4 补充训练 diagnostics 字段：`loss_jepa_global`、`loss_modality_nll`、`top1_acc`、`top5_acc`、`mean_modality_reliability`、`mean_global_reliability`。
- [x] 9.5 补充 focused tests 覆盖随机 output backward、Scenario 32 config 加载和拆分 loss 字段。

## 10. Scenario 32 Missing Mask 配置生效修正

- [x] 10.1 将 `configs/fusion/u_mask_beam_jepa_s32.yaml` 的正式缺失模态配置改为 `missing_mask`，确保训练 extension 实际读取 `p_missing=0.5`。
- [x] 10.2 在 U-MaskBeamJEPA loss config 解析中保留旧 `missing` 字段兼容 warning，并在 `missing` 与 `missing_mask` 同时存在时优先使用 `missing_mask`。
- [x] 10.3 补充 focused config 测试，断言 s32 有效 `missing_mask.p_missing=0.5` 且 smoke 保留原缺失配置。
- [x] 10.4 补充当前实现口径文档，明确 preprocessing 与 simplified encoder 不是严格 JEPA-MSAC。
