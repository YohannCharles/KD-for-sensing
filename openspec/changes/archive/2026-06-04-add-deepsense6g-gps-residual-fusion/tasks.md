## 1. 输入盘点与实现边界

- [x] 1.1 用 inspection spike 盘点 `outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep` 下 r05/r10/r15/r20 的 summary、predictions、support manifest、residual probability 和 figures 文件，记录字段名、缺失字段和可复用路径。
- [x] 1.2 列出本 change 准备新增/修改的源码、配置、测试、README 和 OpenSpec 文件清单，确认不新增顶层 `src.*` 入口。
- [x] 1.3 检查现有 `configs/deepsense6g_gps_adapter_v2.yaml`、`src/kd_sensing/engine/mmw_town_gps_v2.py` 和 DeepSense6G loader，明确 v2 logits 可保存位置和 residual manifest 可复用字段。
- [x] 1.4 新增 `configs/deepsense6g_residual_fusion.yaml`，覆盖 data、residual、modalities、model、loss、train、rerank、experiment、metrics 和 outputs 默认值。

## 2. GPS prior 与 circular residual 工具

- [x] 2.1 扩展 GPS v2 workflow 的配置/CLI，支持显式 `save_logits`，写出 `gps_logits.npy`、`gps_logits_index.csv` 和可选 `gps_prior_probs.npy`，并保持现有 predictions/summary 兼容。
- [x] 2.2 实现 GPS prior artifact loader，按 scene/sample id 对齐 logits index、predictions 和 support/query role，并拒绝重复或缺失映射。
- [x] 2.3 实现 fallback Gaussian prior builder，只从 GPS top1 构造 circular Gaussian prior，默认 sigma 为 `2.0`，并记录 `gps_prior_source=fallback_gaussian_from_top1`。
- [x] 2.4 在现有 circular metrics/loss 相关模块中新增 `signed_circular_residual`、`circular_shift_beam`、`circular_window` 和 GPS good/bad label helper，支持 torch 与 numpy。
- [x] 2.5 更新 `tests/test_circular_metrics.py` 或新增对应测试，覆盖 signed residual wrap-around、shift、window、good/bad threshold 和 fallback prior 不使用 target label。

## 3. Inspection 与 residual manifest

- [x] 3.1 新增包内 inspection CLI 实现，支持 `conda run -n kd_mm_beam python -m kd_sensing.cli.inspect_deepsense6g_residual_inputs --gps-sweep-root outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep --label-space mapping_disabled`。
- [x] 3.2 inspection CLI 自动检查 r05/r10/r15/r20 产物、打印路径与关键字段、报告 GPS logits/probs 是否可用，并输出 prior source 建议。
- [x] 3.3 新增 residual manifest builder，输出 `outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled/manifest/residual_manifest.csv`。
- [x] 3.4 manifest builder 合并 scene、sample id、timestamp/frame id、support/query role、target label、GPS top1/top3/top5、GPS error、signed residual、good/bad label、GPS context feature、prior stats 和 prior source。
- [x] 3.5 manifest builder 自动发现 image/LiDAR/radar path 或 `.npy`、`.npz`、`.pt`、`.csv`、`.parquet` precomputed feature；缺失模态只 warning 并写空列。
- [x] 3.6 新增 `tests/test_residual_manifest.py`，覆盖 manifest 字段、support/query role、fallback prior source、optional modality 缺失不阻断和 GPS context baseline 可运行输入。

## 4. Dataset、encoders 与模型

- [x] 4.1 新增 residual manifest Dataset/DataLoader，只按当前 ablation 启用的 modality 读取字段，`gps_context_only_residual` 不读取 image/LiDAR/radar 文件。
- [x] 4.2 新增 `ImageEncoder`，支持默认 `tiny_cnn` 与可选 `torchvision_resnet18(weights=None)`，不下载 pretrained weights。
- [x] 4.3 新增 `ArrayEncoder`，支持 flat vector、2D map、3D map，并对 shape 不稳定的 raw array 给出清晰错误。
- [x] 4.4 新增 `TabularEncoder`，消费 GPS context、prior stats、pred beam sin/cos 等 tabular feature。
- [x] 4.5 实现 `GPSAnchoredResidualFusion`，输出 `final_logits`、`correction_logits`、`modality_only_logits`、`correction_gate`、`correction_strength` 和 diagnostics。
- [x] 4.6 实现 correction scale 的 softplus 正数参数化、默认初始化 `0.5` 和最大值 clamp `3.0`。
- [x] 4.7 新增 `GPSAnchoredTopKReranker`，构造 GPS top-K、local circular window 和 optional modality top-M 的 union candidate set，并输出 candidate score 与 recall。
- [x] 4.8 新增 `tests/test_residual_fusion_model.py` 和 `tests/test_topk_reranker.py`，覆盖 forward shape、gated formula、correction scale、candidate wrap-around、target 不在 candidate 时 loss mask。

## 5. Loss、训练协议与 ablation

- [x] 5.1 新增 residual loss 模块，组合 final circular soft CE、modality auxiliary CE、gate BCE、good anchor KL 和 correction L2。
- [x] 5.2 实现 hard sample weighting：`gps_error >= good_error_threshold` 的样本按配置增加 final CE 权重。
- [x] 5.3 实现 gate target，仅在 train/support 样本标签上由 GPS good/bad 生成，query label 不参与训练、early stopping 或模型选择。
- [x] 5.4 实现 good anchor loss，只在 `gps_error < good_error_threshold` 的样本上约束 final distribution 接近 GPS prior。
- [x] 5.5 实现 `gps_prior_only`，直接复现 GPS v2 r15 baseline 并写入 residual summary。
- [x] 5.6 实现 `target_adapt_beambench_residual`：source pretrain、target support finetune、target query final evaluation；source prior 不完整时降级 `support_only`。
- [x] 5.7 实现 `within_scene_residual_upper_bound`，并在 summary/report 中标记为 sanity upper bound，禁止进入主结论。
- [x] 5.8 实现默认 ablation：`gps_prior_only`、`gps_context_only_residual`、`gps_plus_residual_no_gate`、`gps_plus_residual_gated`、`gps_plus_residual_gated_anchor`、`gps_topk_rerank`。
- [x] 5.9 实现 optional modality ablation 自动启用/跳过：image、LiDAR、radar、all available，并在 summary 写入 modalities 或 `skipped_reason`。
- [x] 5.10 新增 `tests/test_residual_losses.py`，覆盖 hard sample weighting、good anchor mask、gate target、query leakage guard 和 finite loss。

## 6. 输出、可视化与对比报告

- [x] 6.1 新增 residual train/eval 包内 CLI，支持 `conda run -n kd_mm_beam python -m kd_sensing.cli.run_deepsense6g_residual_fusion --config configs/deepsense6g_residual_fusion.yaml --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 6.2 写出 `summary_overall.csv`、`summary_by_scene.csv`、`summary_by_gps_good_bad.csv`，字段包含 GPS baseline、residual 指标、delta、good degradation 和 bad correction。
- [x] 6.3 写出 `predictions.csv`、`correction_events.csv` 和 `candidate_recall.csv`，每行包含 scene、sample id、support/query role、GPS/final prediction、circular error、gate、delta、prior stats、modalities 和 train mode。
- [x] 6.4 新增 plot CLI，生成 ENU scatter、before/after residual histogram、signed residual before/after、gate diagnostics、good/bad bar、label distribution 和可选 modality montage/heatmap。
- [x] 6.5 新增 compare CLI，读取 GPS v2 r15/r20 summary 与 residual summary，写出 `comparison_report.md`，自动回答是否超过 r15、是否接近 r20、scene 贡献、hard correction、good degradation、gate 行为和多模态收益。
- [x] 6.6 确保所有输出 metadata 记录 prior source、train mode、support/query count、query label leakage guard、enabled/skipped modalities 和 result directory。

## 7. 文档、入口与架构检查

- [x] 7.1 更新 README，新增 “DeepSense6G residual correction after GPS v2” 章节，说明 GPS v2 现有结论、gate/good anchor loss 原因、运行命令和结果判读。
- [x] 7.2 如需新增 console scripts，更新 `pyproject.toml` 并确保它们委托 `src/kd_sensing/cli/` 包内实现。
- [x] 7.3 更新 docs 或 project surface inventory，说明新增 residual workflow 属于 GPS-prior anchored residual correction，不是从零多模态 beam prediction。
- [x] 7.4 更新架构边界测试，拒绝新增顶层 `src.*` residual 入口，确认包内 CLI import 不触发训练或大型数据读取。

## 8. 验证与验收

- [x] 8.1 运行 `openspec validate add-deepsense6g-gps-residual-fusion --strict`。
- [x] 8.2 运行 `conda run -n kd_mm_beam pytest tests/test_residual_manifest.py tests/test_residual_fusion_model.py tests/test_residual_losses.py tests/test_topk_reranker.py tests/test_circular_metrics.py -q`。
- [x] 8.3 运行 `conda run -n kd_mm_beam pytest tests/test_mmw_town_gps_adapter_v2.py tests/test_circular_metrics.py -q`，确认 GPS v2 与 circular metrics 回归未破坏。
- [x] 8.4 运行 inspection 命令：`conda run -n kd_mm_beam python -m kd_sensing.cli.inspect_deepsense6g_residual_inputs --gps-sweep-root outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep --label-space mapping_disabled`。
- [x] 8.5 运行 manifest 命令：`conda run -n kd_mm_beam python -m kd_sensing.cli.prepare_deepsense6g_residual_manifest --config configs/deepsense6g_residual_fusion.yaml --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 8.6 运行 residual train/eval 命令：`conda run -n kd_mm_beam python -m kd_sensing.cli.run_deepsense6g_residual_fusion --config configs/deepsense6g_residual_fusion.yaml --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 8.7 运行 plot 命令：`conda run -n kd_mm_beam python -m kd_sensing.cli.plot_deepsense6g_residual_fusion --results-dir outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled`。
- [x] 8.8 运行 compare 命令：`conda run -n kd_mm_beam python -m kd_sensing.cli.compare_deepsense6g_residual_with_gps_v2 --gps-v2-root outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep --residual-root outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled --support-ratio 0.15 --label-space mapping_disabled`。
- [x] 8.9 检查 `gps_prior_only` 是否在容差内复现 v2 r15 的 `DBA≈0.6707`、`mean err≈2.344`、`P(error<4)≈0.8523`；若不一致，在 report 中记录原因。
- [x] 8.10 最终运行 `conda run -n kd_mm_beam pytest -q`，并记录任何因本地数据、外部依赖或环境缺失导致无法完成的验证。
