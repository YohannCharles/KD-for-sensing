## 1. 后端 schema 与预测来源

- [x] 1.1 扩展 `canonical_subset_name()` 或新增解析 helper，支持 `single_best_mmwave`、`teacher_mmwave`、`teacher_gps`、`gps_only`、`mmwave_only` 等强势模态别名。
- [x] 1.2 新增通用单模态预测来源选择 helper，支持从 `subset_predictions` 和 `teacher_predictions` 解析指定 modality，并返回 source metadata、row count 和不可用原因。
- [x] 1.3 保留现有 weak prediction 选择入口，并让它复用新的通用 helper 或保持兼容包装，确保旧测试不需要改调用方式。
- [x] 1.4 定义 pair mode metadata 结构，记录 `strong_modalities`、`strong_prediction_sources`、`fusion_subset_availability`、unmatched pair 和 warnings。

## 2. 强弱互补 case mining

- [x] 2.1 扩展 `build_case_table()` 参数，新增可选 `strong_modalities` 和 pair-level fusion subset mapping，同时保持旧 `strong_subset` 默认行为不变。
- [x] 2.2 在 pair mode 中按 `strong_modality × weak_modality × horizon × sample` 对齐 strong、weak 和可选 fusion prediction。
- [x] 2.3 为 pair mode case 表写出 `strong_modality`、`strong_weak_pair`、`strong_prediction_source`、`weak_prediction_source`、`fusion_prediction_available` 和 resolved subset/source 字段。
- [x] 2.4 在 fusion subset 可用时计算 `fusion_pred`、`fusion_correct`、`p_true_fusion`、`fusion_gt_gain` 和现有 rescue / unused / negative transfer 标签。
- [x] 2.5 在 fusion subset 缺失时仍输出 strong/weak 互补 case，并让 fusion-dependent 字段、case type 或指标按不可用语义安全降级。
- [x] 2.6 确认未启用 `strong_modalities` 的旧路径仍输出当前 `strong_only`、`weak_modality`、`fusion_subset`、case type 和 summary。

## 3. Summary、bucket 与报告

- [x] 3.1 扩展 `compute_summary()`，当 case 表包含 `strong_modality` 时输出 `by_strong_modality`。
- [x] 3.2 扩展 `compute_summary()`，当 case 表包含 `strong_weak_pair` 时输出 `by_strong_weak_pair`，包含 count、horizon count、complementarity rate、mean weak gain 和 fusion availability。
- [x] 3.3 调整 `_summary_metrics()` 或相关 helper，让 fusion unavailable 的 group 对 rescue、unused complementary、negative transfer 和 fusion gain 输出空值或不可用原因。
- [x] 3.4 更新 `render_report()`，在存在 strong modality pair 分析时展示强势模态维度、最佳 strong/weak pair 和 fusion availability 摘要。

## 4. CLI 与文档入口

- [x] 4.1 扩展 `scripts/analysis/build_complementarity_cases.py`，新增 `--strong-modalities` 参数，默认支持 Scene32 的 `gps,mmwave` 或配置覆盖。
- [x] 4.2 新增 `--pair-fusion-subsets` 参数，支持 `strong+weak=subset` 形式，并保留现有 `--fusion-subsets` 的旧 strong-only 语义。
- [x] 4.3 在 CLI 日志中输出 strong prediction sources、fusion availability、unmatched pair 和降级 warning。
- [x] 4.4 更新 `tools/visualization/README.md`，补充“选择强势模态 + 一个或全部 weak modality”的命令示例和 Explorer 筛选示例。

## 5. Explorer helper 与 Gradio UI

- [x] 5.1 扩展 `build_complementarity_choices()`，在 case 表包含 `strong_modality` 时返回 `strong_modalities` 和默认值，旧 case 表退化为 `all`。
- [x] 5.2 扩展 `filter_complementarity_cases()`，新增 `strong_modality` 筛选参数，并确保 `Strong Modality=all`、`Weak Modality=all` 能返回全部 pair。
- [x] 5.3 扩展 `case_detail_payload()`，展示 `strong_modality`、`strong_prediction_source`、`strong_weak_pair` 和 fusion availability。
- [x] 5.4 扩展 `export_filtered_cases()` 路径覆盖，确保导出的 filtered CSV 保留 strong modality 新字段。
- [x] 5.5 在 `tools/visualization/gradio_multimodal_viewer.py` 的 Complementarity Explorer Tab 中新增 `Strong Modality` Dropdown，并接入 apply filters 回调。
- [x] 5.6 确认旧 complementarity 输出缺少 `strong_modality` 时，Gradio 页面仍能加载、筛选、导出和点击样本。

## 6. 自动化测试

- [x] 6.1 扩展 `tests/test_complementarity_analysis.py`，构造包含 `gps`、`mmwave`、`image`、`radar`、`lidar` teacher predictions 的小型输入，覆盖指定 strong modality 与多个 weak modalities。
- [x] 6.2 增加缺失 pair fusion subset 的降级测试，验证 case 表保留 strong/weak 互补行，fusion 字段和 summary 按不可用语义输出。
- [x] 6.3 增加 fusion subset 可用测试，验证 `fusion_prediction_available=true` 时 rescue、unused complementary、negative transfer 和 fusion gain 正常计算。
- [x] 6.4 扩展 summary 测试，验证 `by_strong_modality`、`by_strong_weak_pair`、prediction source metadata 和 fusion availability。
- [x] 6.5 扩展 `tests/test_gradio_complementarity_explorer.py`，覆盖 strong modality choices、默认值、单 strong 筛选、strong all + weak all、导出字段和详情 JSON。
- [x] 6.6 使用 `conda run -n kd_mm_beam pytest tests/test_complementarity_analysis.py tests/test_gradio_complementarity_explorer.py` 验证新增和既有互补分析测试。

## 7. 回归与验收

- [x] 7.1 使用旧参数运行 `conda run -n kd_mm_beam python scripts/analysis/build_complementarity_cases.py --scene scene32 --input-path outputs/scene32/scene32_marf/conditional_utility --output-dir outputs/scene32/complementarity_analysis`，确认旧 strong-only 输出兼容。
- [x] 7.2 使用新参数运行 `conda run -n kd_mm_beam python scripts/analysis/build_complementarity_cases.py --scene scene32 --input-path outputs/scene32/scene32_marf/conditional_utility --output-dir outputs/scene32/complementarity_analysis_strong_pairs --strong-modalities mmwave --weak-modalities image,radar,lidar`，确认能生成 strong modality pair case。
- [x] 7.3 使用 `conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py --manifest tools/visualization/sample_manifest_example.json --complementarity-dir outputs/scene32/complementarity_analysis_strong_pairs --check-only` 验证 viewer 可加载新输出。
- [x] 7.4 使用 `openspec status --change add-strong-modality-complementarity-explorer` 确认 proposal、design、specs 和 tasks 都达到可实施状态。
