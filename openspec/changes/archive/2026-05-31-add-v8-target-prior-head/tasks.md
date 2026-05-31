## 1. 配置与模型结构

- [x] 1.1 在 HiST-Beam variant 注册/解析中加入 `v8_target_prior_head`，并解析 `hist_beam.v8` 默认配置。
- [x] 1.2 在 `HistBeamConfig` 中增加 v8 参数：mode、adapter_dim、adapter_dropout、use_adapter、use_target_prior、use_source_logits_in_final、lambda_src、lambda_tgt、beta_prior、learnable_beta_prior、use_coarse_to_fine、sector_size、unfreeze_last_fusion_block。
- [x] 1.3 在 `HistBeamFusionNet` 中新增 `target_adapter`、`target_head`、`target_prior_bias`、可选 `beta_prior`、可选 `sector_head` 和 `offset_head`。
- [x] 1.4 实现 v8 forward 分支，确保 `logits`、`beam_logits`、`logits_final`、`target_logits`、`source_logits`、`target_prior_bias`、`features` 和 `hist_beam` metadata 输出兼容现有 engine。
- [x] 1.5 添加 `set_target_prior_from_labels(...)` 方法，并确保空 support labels fallback 为 uniform prior。

## 2. Prior、Soft Label 与 Loss

- [x] 2.1 实现 `gaussian_smooth_beam_prior(labels, num_beams, sigma=1.5, eps=1e-4, device=None)`，只消费传入的 support labels。
- [x] 2.2 实现 `make_beam_soft_labels(labels, num_beams, sigma=1.0, eps=1e-8)`，并覆盖 shape、归一化和 ignore index 安全处理。
- [x] 2.3 在 `compute_hist_beam_loss` 中新增 v8 检测和 `_compute_v8_hist_beam_loss` 分支。
- [x] 2.4 实现 v8 final soft CE、hard CE fallback、prior smoothness loss、sector CE 和 offset CE，并输出 `hist/v8/*` diagnostics。
- [x] 2.5 确保 v8 target adaptation loss 不读取 target-side beam_power、CSI、path 或 radio fields。

## 3. Target Adaptation 与 Freeze Policy

- [x] 3.1 在 `apply_hist_beam_adaptation_strategy` 中加入 `v8_target_head_only`，冻结 source backbone/source heads，只训练 v8 target branch 和允许参数。
- [x] 3.2 在 target adaptation stage few-shot sampling 后，用 sampled labeled target_adapt labels 调用 `model.set_target_prior_from_labels(...)`。
- [x] 3.3 将 v8 support label hist、smoothed prior top beams、prior bias top beams、trainable parameter names/summary 写入 metrics、adapt log 或独立 artifact。
- [x] 3.4 支持 v8 A2-A5 模式配置：`target_linear_probe`、`target_prior_head`、`source_prior_only`、`target_prior_coarse_to_fine`。
- [x] 3.5 为 `run_prototype_probe=true` 预留 evaluation-only prototype probe 接口；若暂不实现，写出 `prototype_probe_available=false` 和 unavailable reason。

## 4. 评估产物与配置

- [x] 4.1 在 source-only target eval 和 adapted target eval 后生成 `prediction_hist.json`，包含 true/pred hist、top beams、mean absolute beam error、within-1/2/3 accuracy。
- [x] 4.2 将 prediction histogram artifact path 和关键字段接入 run metrics 或 LOSO summary。
- [x] 4.3 新增 `configs/hist_beam/v8_target_prior_head.yaml` 或 quick validation override 示例，默认不影响 v0-v7 配置。
- [x] 4.4 预留 `source_train.loss_type`、`class_prior_from_source_train` 和 `logit_adjust_tau` 配置入口；若未实现去偏 loss，必须清晰记录 unsupported reason。

## 5. 测试与验证

- [x] 5.1 添加模型单元测试：v8 构建、forward 输出键、默认 final logits 不含 source logits、source logits 融合 opt-in。
- [x] 5.2 添加 prior/soft label 单元测试：Gaussian prior top beams、uniform fallback、soft label 归一化、防止使用 target_test 输入。
- [x] 5.3 添加 loss 单元测试：v8 soft CE、hard CE fallback、prior smoothness、coarse-to-fine loss 和最后 sector 安全处理。
- [x] 5.4 添加 freeze policy/adaptation 测试：`v8_target_head_only` trainable 参数集合、trainable ratio、prior 初始化只来自 sampled target_adapt labels。
- [x] 5.5 添加 histogram artifact 测试：source-only 和 adapted eval 写出 `prediction_hist.json` 且字段完整。
- [x] 5.6 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 5.7 运行 v8 相关定向测试，例如 `conda run -n kd_mm_beam pytest <v8-test-files> -q`。
- [x] 5.8 运行 `openspec validate add-v8-target-prior-head --strict`。
