## 1. Subset 与 Metadata 基础

- [x] 1.1 新增 `src/kd_sensing/evaluation/subset_specs.py`，定义 `SCENE32_CONDITIONAL_UTILITY_SUBSETS`，并用 `normalize_modalities()` 保证中心模态顺序。
- [x] 1.2 修改 `src/kd_sensing/engine/validator.py`，让 `evaluation.modality_subsets` 解析 conditional audit subset 时复用 subset registry，并保持 `top_prior`、`single_best_prior` 等既有名称兼容。
- [x] 1.3 为 `DeepSense6GDataset` 增加 opt-in metadata 返回能力，至少包含 `dataset_index`、稳定 `sample_id`，并在可可靠解析时返回 `seq_id`、`frame_idx` 或路径追踪字段。
- [x] 1.4 更新或新增分析配置，使 audit dataloader 启用 metadata，但现有训练配置默认不启用。

## 2. 逐样本指标与 Dump

- [x] 2.1 新增 `src/kd_sensing/diagnostics/conditional_utility.py` 的 logits-to-record helper，输出 top-k prediction/probability、gt probability、CE、Top1/Top3/Top5 hit、beam distance 和 DBA contribution。
- [x] 2.2 实现 parquet 优先、`csv.gz` fallback 的表格写入 helper，并在 metadata 中记录实际输出格式。
- [x] 2.3 实现 `subset_predictions` 生成逻辑，按 `sample_id + horizon + subset` 写出逐样本记录。
- [x] 2.4 实现 `conditional_utility_per_sample_delta` 计算，以 `strong_only` 对比 `strong_plus_image/radar/lidar` 输出 CE、Top1、Top3 和 DBA delta。
- [x] 2.5 增加聚合 helper，将逐样本记录汇总为与现有 `metrics.json` 语义一致的 Top1、Top3、Top5、DBA 和 loss/CE 指标。

## 3. Audit Runner 与 Teacher Complementarity

- [x] 3.1 新增 `tools/analysis/run_conditional_utility_audit.py`，复用现有 config、checkpoint、normalization artifact、model、criterion 和 dataloader 构建路径。
- [x] 3.2 在 runner 中校验模型支持 `force_modality_mask`，否则输出清晰错误并停止。
- [x] 3.3 实现 subset forward 循环，使用 registry subset mask 对同一 MARF checkpoint 评估 `all`、`strong_only`、`strong_plus_*`、`single_best_mmwave` 和 `weak_only`。
- [x] 3.4 复用或扩展 `TeacherEnsemble`，从配置的 teacher registry 严格加载单模态 teacher，确保 teacher 为 `eval()`、`requires_grad=False`、`torch.no_grad()`，且 logits 为 `[B, 3, 64]`。
- [x] 3.5 实现 `teacher_predictions` dump 与 `teacher_complementarity_summary.json`，统计 rescue rate、gt probability advantage rate 和 CE better rate。
- [x] 3.6 实现 `oracle_subset_summary.json`，按每个样本和 horizon 的最小 CE 选择 oracle subset，并输出 oracle 指标和选择分布。

## 4. 通信状态 Bucket 与 Summary

- [x] 4.1 新增 `src/kd_sensing/diagnostics/communication_state_features.py`，计算 mmWave entropy、top1 prob、top1-top2 margin、peak sharpness、total power 和 peak drift。
- [x] 4.2 在同一模块中计算 GPS range、bearing、delta range、delta bearing、angular velocity、gps jump magnitude，以及 beam transition indicator。
- [x] 4.3 实现基于验证集分位数的 bucket 分配，并支持 low/mid/high、low/high、transition/stable 等 bucket 名称。
- [x] 4.4 生成 `conditional_utility_by_bucket.csv`，按 bucket、weak modality 和 horizon 输出 delta Top1、Top3、DBA、CE、oracle choice rate 和 teacher rescue rate。
- [x] 4.5 生成 `conditional_utility_summary.json`，包含 aggregate metrics、marginal utility、oracle、teacher complementarity、bucket highlights、metadata 和阈值配置。
- [x] 4.6 实现 diagnosis 规则，输出 `global_useful`、`conditionally_useful`、`representation_exists_but_not_exploited` 或 `currently_low_utility`，并记录触发依据。

## 5. 配置与图表

- [x] 5.1 新增 `configs/analysis/scene32_marf_conditional_utility_audit.yaml`，配置 MARF checkpoint、teacher registry、subset、horizon、输出目录、bucket features 和 diagnosis 阈值。
- [x] 5.2 新增 `tools/analysis/analyze_conditional_utility.py`，从 audit 输出读取 summary、delta、bucket 和 teacher 表格。
- [x] 5.3 生成 `subset_metrics_bar.png`、`marginal_delta_by_horizon.png`、`oracle_choice_distribution.png`、`teacher_rescue_rate.png`、`delta_ce_histogram_<modality>.png` 和 `bucket_heatmap_delta_dba.png`。
- [x] 5.4 确保绘图脚本将图输出到 `conditional_utility/figures/`，不覆盖既有可视化工具输出。

## 6. 测试与验证

- [x] 6.1 新增 `tests/test_subset_specs.py`，验证七个 conditional audit subset 存在、名称正确且模态顺序合法。
- [x] 6.2 新增 `tests/test_conditional_utility_metrics.py`，用 toy logits/labels 验证 CE、Top-k hit、DBA contribution 和 aggregate DBA 一致性。
- [x] 6.3 新增 `tests/test_conditional_utility_oracle.py`，验证 subset oracle 和 teacher rescue/complementarity 规则。
- [x] 6.4 新增 `tests/test_communication_state_features.py`，验证 mmWave、GPS 和 beam transition 特征 shape 与关键数值。
- [x] 6.5 新增 dummy end-to-end audit 测试，使用小批量 synthetic 或 toy logits 生成 subset predictions、teacher predictions、summary、bucket CSV 和 oracle summary。
- [x] 6.6 运行 `conda run -n kd_mm_beam pytest -q tests/test_subset_specs.py tests/test_conditional_utility_metrics.py tests/test_conditional_utility_oracle.py tests/test_communication_state_features.py`。
- [x] 6.7 运行 `conda run -n kd_mm_beam pytest -q`。
- [x] 6.8 运行 `openspec validate add-conditional-utility-audit --strict`，确认 proposal、spec 和 tasks 合法。
