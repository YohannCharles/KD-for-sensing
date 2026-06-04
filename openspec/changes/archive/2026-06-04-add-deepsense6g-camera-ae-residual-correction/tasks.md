## 1. 输入盘点与配置边界

- [x] 1.1 盘点现有 `configs/deepsense6g_residual_fusion.yaml`、`src/kd_sensing/data/deepsense6g_residual.py`、`src/kd_sensing/engine/deepsense6g_residual_fusion.py` 和 GPS v2 r15 输出字段，记录可复用 loader、metric、summary 和 fallback prior 逻辑。
- [x] 1.2 新增 `configs/deepsense6g_camera_residual.yaml`，覆盖 data、image、ae、residual、model、loss、train、rerank、experiment、metrics 和 outputs 默认值。
- [x] 1.3 确认新增入口全部位于 `src/kd_sensing/cli/`，不新增顶层 `src.*` 运行入口或兼容包装脚本。
- [x] 1.4 明确本 change 输出根目录为 `outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled/`，训练与 feature 产物分别写入 `outputs/training/` 和 `outputs/features/`。

## 2. Camera residual manifest 与 Dataset

- [x] 2.1 新增 `src/kd_sensing/data/deepsense6g_camera_residual.py`，复用 GPS v2 artifact discovery，自动读取 predictions、top-K、support/query role、GPS error、signed residual、GPS context 和 GPS prior logits。
- [x] 2.2 实现 GPS prior loader：优先读取 `gps_logits.npy`、`logits.npy`、`pred_logits.npy` 和 logits index；缺失时只用 `gps_pred_top1` 构造 circular Gaussian fallback，并记录 `gps_prior_source`。
- [x] 2.3 实现 image path 自动发现，支持 DeepSense6G 常见 image/camera/rgb 路径结构；找不到 image 时保留样本并写入 `image_exists=false`。
- [x] 2.4 输出 `camera_residual_manifest.csv` 和 manifest metadata，字段覆盖 scene、sample id、timestamp/frame id、split role、target、GPS top-K、GPS residual/good-bad、GPS prior stats、image path 和 AE feature placeholder。
- [x] 2.5 实现 camera residual Dataset/DataLoader：`gps_prior_only` 不读取 image/feature；AE training 只读取 `image_exists=true`；residual training 按 ablation 读取 AE feature 或降级为 GPS context。
- [x] 2.6 新增 `src/kd_sensing/cli/prepare_deepsense6g_camera_residual_manifest.py`，支持 `--config`、`--support-ratio`、`--label-space`，并可用 `conda run -n kd_mm_beam python -m kd_sensing.cli.prepare_deepsense6g_camera_residual_manifest --help` 检查。

## 3. Circular residual delta 工具

- [x] 3.1 在现有 circular/residual utility 中新增或复用 `signed_circular_residual(target, pred, num_beams)`，覆盖 `target=1,pred=63 -> 2` 和 `target=63,pred=1 -> -2`。
- [x] 3.2 实现 `residual_to_delta_class(residual, radius)`，将 `[-R, R]` 映射到稳定 class id，并将 `abs(residual)>R` 映射到 overflow class `2R+1`。
- [x] 3.3 实现 `delta_class_to_residual(class_id, radius)`，local class 返回 residual，overflow 返回 `None` 或配置声明的 special value。
- [x] 3.4 扩展 diagnostics/metadata，统计 residual overflow count、local residual coverage 和 GPS hard/good 分组数量。

## 4. Stage A: Camera AE

- [x] 4.1 新增 `src/kd_sensing/models/camera_autoencoder.py`，实现 tiny convolutional `CameraAutoEncoder`，默认 `latent_dim=128`，不下载 pretrained weights。
- [x] 4.2 实现 AE image transform、normalization、train/val split、MSE reconstruction loss、early stopping、resume 和 reconstruction example 保存。
- [x] 4.3 默认 AE 训练使用 source scenes 全部可用 image 与 target support image；实现 `ae.use_target_query_unlabeled` 配置，默认 `false`，开启时 metadata 记录 transductive 设置但不使用 query label。
- [x] 4.4 新增 `src/kd_sensing/cli/train_deepsense6g_camera_ae.py`，保存 `checkpoints/best.pt`、`metrics.csv` 和 `recon_examples/`。
- [x] 4.5 AE 训练集没有任何可用 image 时抛出清晰错误，不写出成功 checkpoint。

## 5. Stage A 输出: AE feature extraction

- [x] 5.1 新增 `src/kd_sensing/cli/extract_deepsense6g_camera_ae_features.py`，加载 AE checkpoint 并导出 frozen encoder latent feature。
- [x] 5.2 输出 `outputs/features/deepsense6g_camera_ae/r15/mapping_disabled/features.npy` 和 `features_index.csv`，保证 shape 与 index 行数一致。
- [x] 5.3 生成 `camera_residual_manifest_with_ae.csv`，新增 `ae_feature_row_index` 和 `ae_feature_path`，保留 feature 不可用样本并写明原因。
- [x] 5.4 支持 feature extraction resume 或跳过已存在且 fingerprint 匹配的 feature 文件。

## 6. Stage B: Residual/Gate 模型与 loss

- [x] 6.1 新增 `src/kd_sensing/models/camera_residual_fusion.py`，实现 `CameraGPSResidualFusion`，输入 GPS prior logits、GPS pred top1、GPS context 和 AE feature。
- [x] 6.2 模型输出 `residual_delta_logits`、`correction_gate`、`p_corr`、`final_logits` 和 diagnostics；默认 `delta_radius=8`，gate bias 初始化为 `-2.0`。
- [x] 6.3 实现 `p_corr` 合成：local delta 映射到 `(gps_pred_top1 + delta) mod 64`，overflow 按配置均匀分配或忽略，概率归一化稳定。
- [x] 6.4 实现 gated final distribution：`p_final=(1-gate)*p_gps + gate*p_corr`，并输出 `log(p_final + eps)`。
- [x] 6.5 新增 `src/kd_sensing/losses/camera_residual_losses.py`，实现 final circular soft CE、residual delta CE、gate BCE、good-anchor KL、optional aux CE 和 optional gate entropy regularization。
- [x] 6.6 hard samples 使用 `gps_error >= 4` 加权；good-anchor 只作用于 `gps_error < 4`；query label 不参与 residual delta/gate/early stopping/model selection。
- [x] 6.7 支持 image feature 缺失时的 `gps_context_only_residual` 路径，并在 summary 记录降级或跳过原因。

## 7. Stage B: 训练、评估与 ablation

- [x] 7.1 新增 `src/kd_sensing/engine/deepsense6g_camera_residual.py`，实现 `target_adapt_beambench_camera_residual`，按 target scene 循环处理 source pretrain、target support finetune 和 target query final evaluation。
- [x] 7.2 实现 train modes：`support_only`、`source_pretrain_target_finetune` 和 `source_plus_support`，默认 `source_pretrain_target_finetune`。
- [x] 7.3 实现默认 ablation：`gps_prior_only`、`gps_context_only_residual`、`camera_ae_only_direct_beam`、`camera_ae_plus_gps_concat_direct_beam`、`camera_ae_residual_gated`、`camera_ae_residual_gated_anchor` 和 `camera_ae_residual_gated_anchor_source_pretrain`。
- [x] 7.4 `gps_prior_only` 必须在配置容差内复现 GPS v2 r15 的 `DBA≈0.6707`、`mean err≈2.344`、`P(error<4)≈0.8523`，不一致时写入 comparison report。
- [x] 7.5 新增 `src/kd_sensing/cli/run_deepsense6g_camera_residual.py`，支持 `--config`、`--support-ratio`、`--label-space`，输出 checkpoints、metrics、summary 和 predictions。

## 8. Stage C: Candidate attention reranker

- [x] 8.1 新增 `src/kd_sensing/models/beam_candidate_attention.py`，实现最小 `BeamCandidateAttentionReranker`。
- [x] 8.2 candidate set 使用 GPS top-K 与 GPS top1 local circular window 的 union，默认 `gps_topk=16`、`local_radius=8`，beam id 必须 wrap 到 `[0, 64)`。
- [x] 8.3 使用 AE feature 作为 pseudo image token，并兼容后续 `[B, N, D]` patch tokens。
- [x] 8.4 输出 candidate scores、rerank top1/top3、`target_in_gps_top16`、`target_in_local_radius8` 和 `target_in_union_candidates`。
- [x] 8.5 将 `camera_ae_query_rerank` 标记为 optional ablation，不作为 Stage A/B 验收性能硬门槛。

## 9. 输出、可视化与对比报告

- [x] 9.1 写出 `summary_overall.csv`、`summary_by_scene.csv` 和 `summary_by_gps_good_bad.csv`，字段包含 GPS baseline、final 指标、delta、good degradation、bad correction、gate AUC/均值和 train mode。
- [x] 9.2 写出 `predictions.csv`、`correction_events.csv` 和可选 `candidate_recall.csv`，包含 scene、sample id、split role、target、GPS/final pred、errors、residual delta、gate、image/AE metadata、ablation 和 train mode。
- [x] 9.3 新增 `src/kd_sensing/cli/plot_deepsense6g_camera_residual.py`，生成 ENU scatter、improvement、residual histogram、signed residual、gate diagnostics、good/bad bar、label distribution、delta confusion matrix 和 image correction montage。
- [x] 9.4 新增 `src/kd_sensing/cli/compare_deepsense6g_camera_residual_with_gps_v2.py`，读取 GPS v2 r15/r20 summary 与 camera residual summary，写出 `comparison_report.md`。
- [x] 9.5 run metadata 记录 prior source、AE checkpoint、feature fingerprint、model selection split、query label usage、support/query count、target scene、ablation 和 skipped reasons。

## 10. 文档、测试与验证

- [x] 10.1 更新 README 或 docs，新增 DeepSense6G camera residual after GPS v2 章节，说明 GPS prior frozen、Camera AE、residual/gate、good-anchor、ablation 和判读方式。
- [x] 10.2 新增 `tests/test_camera_ae.py`、`tests/test_camera_residual_fusion.py`、`tests/test_camera_residual_losses.py`、`tests/test_beam_candidate_attention.py` 和 manifest/query leakage 相关测试。
- [x] 10.3 运行 `openspec validate add-deepsense6g-camera-ae-residual-correction --strict`。
- [x] 10.4 运行 `conda run -n kd_mm_beam pytest tests/test_camera_ae.py tests/test_camera_residual_fusion.py tests/test_camera_residual_losses.py tests/test_beam_candidate_attention.py -q`。
- [x] 10.5 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_circular_metrics.py tests/test_residual_manifest.py tests/test_residual_fusion_model.py tests/test_residual_losses.py tests/test_topk_reranker.py -q`，确认 GPS v2 与已有 residual workflow 未回归。
- [x] 10.6 运行 manifest smoke：`conda run -n kd_mm_beam python -m kd_sensing.cli.prepare_deepsense6g_camera_residual_manifest --config configs/deepsense6g_camera_residual.yaml --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 10.7 如本地 image 可用，运行 AE/feature/residual 最小 smoke；如 image 不可用，在验证记录中说明无法完成的原因和 manifest 的 image availability。
- [x] 10.8 最终按风险运行 `conda run -n kd_mm_beam pytest -q` 或记录因本地数据/环境限制无法运行的部分。
