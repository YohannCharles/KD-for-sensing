## 1. 后端数据与 schema 适配

- [x] 1.1 新增 `src/kd_sensing/diagnostics/complementarity.py`，定义读取表、schema adapter、subset 名称映射和缺失能力 metadata 的核心数据结构。
- [x] 1.2 实现 `load_subset_predictions()`，支持从目录或文件读取 `subset_predictions`、`teacher_predictions`、`conditional_utility_per_sample_delta`、`communication_state_features` 和 `conditional_utility_by_bucket`。
- [x] 1.3 实现 `normalize_schema()`，把当前字段 `gt_beam`、`pred_top1`、`gt_prob`、`top1_prob`、`top2_prob` 等映射为内部标准列。
- [x] 1.4 实现 strong subset、weak modality 和 fusion subset 的别名解析，默认支持 `strong_only`、`image/radar/lidar`、`strong_plus_image/radar/lidar` 以及 `gps+mmwave` 风格别名。
- [x] 1.5 实现弱模态预测来源选择：优先单弱模态 subset，其次 `teacher_predictions`，缺失时输出 `weak_prediction_available=false` 与限制说明。

## 2. case mining 与统计输出

- [x] 2.1 实现 `build_case_table()`，按 `sample_id × dataset_index × horizon × weak_modality` 对齐 strong、weak 和 fusion 预测。
- [x] 2.2 实现 top1 正确性、互斥 `case_type`、多标签 `research_tags`、`weak_prediction_source` 和 unmatched metadata 输出。
- [x] 2.3 实现概率指标计算：`p_true_*`、`*_margin`、`weak_gt_gain`、`fusion_gt_gain`，并在概率字段缺失时安全输出空值。
- [x] 2.4 合并 `conditional_utility_per_sample_delta` 中已有的 `delta_ce`、`delta_top1`、`delta_top3` 和 `delta_dba` 字段。
- [x] 2.5 实现 `compute_summary()`，输出全局、按 weak modality、按 horizon 和按 case type 的 count、rate、分子分母、net fusion gain 与概率均值。
- [x] 2.6 实现 `compute_bucket_summary()`，优先使用逐样本 bucket 字段或 `communication_state_features`，不可用时记录降级原因。
- [x] 2.7 实现 `write_outputs()`，写出 `complementarity_cases.csv.gz`、`complementarity_summary.json`、`complementarity_by_bucket.csv` 和 `complementarity_report.md`。

## 3. 命令入口与运行验证

- [x] 3.1 新增 `scripts/analysis/build_complementarity_cases.py`，提供 `--scene`、`--input-path`、`--output-dir`、`--strong-subset`、`--weak-modalities`、`--fusion-subsets` 和 `--horizons` 参数。
- [x] 3.2 在脚本日志中输出输入字段、schema mapping、弱模态预测来源、概率指标可用性、输出行数和降级 warning。
- [x] 3.3 使用 `conda run -n kd_mm_beam python scripts/analysis/build_complementarity_cases.py --scene scene32 --input-path outputs/scene32/scene32_marf/conditional_utility --output-dir outputs/scene32/complementarity_analysis` 验证当前 Scene32 产物可生成分析输出。
- [x] 3.4 检查生成的 summary metadata，确认当前 `subset_predictions.csv.gz` 的字段、weak prediction source 和 probability metrics 状态被正确记录。

## 4. Gradio Explorer 集成

- [x] 4.1 为 `tools/visualization/gradio_multimodal_viewer.py` 增加 `--complementarity-dir` 参数，并实现缺省空状态加载逻辑。
- [x] 4.2 新增 viewer helper，用于加载 complementarity case 表、summary、bucket 表，并构造 scene/horizon/weak/case/bucket/sort 控件 choices。
- [x] 4.3 新增可测试的 Explorer 筛选函数，支持 scene、horizon、weak modality、case type、bucket、min gain、排序和最大展示行数。
- [x] 4.4 在 Gradio 页面新增 `Complementarity Explorer` Tab，包含筛选区、统计面板、case type 图、bucket 图、样本表、导出按钮和固定研究解释文本。
- [x] 4.5 实现 Dataframe 行选择联动，根据 `sample_id` 或 `dataset_index` 定位 manifest 样本，并复用现有 sample render 逻辑展示 raw/processed modalities 与 diagnostics。
- [x] 4.6 实现 manifest 无法匹配或完整分布不可用时的安全空状态与提示。
- [x] 4.7 实现 Export filtered CSV 回调，写出当前筛选结果并返回 Gradio 可下载文件。

## 5. 测试

- [x] 5.1 新增后端 case 判定单元测试，覆盖 rescue、unused complementary、negative transfer、strong wrong fusion correct、all correct、all wrong 和 other。
- [x] 5.2 新增 summary 指标测试，验证 `complementarity_rate`、`rescue_rate_given_complementary`、`unused_complementary_rate`、`negative_transfer_rate` 和 `net_fusion_gain_count`。
- [x] 5.3 新增 schema adapter 测试，覆盖 subset 别名、teacher prediction fallback 和缺失 probability 字段。
- [x] 5.4 新增 bucket 降级测试，覆盖有逐样本 bucket 和无 bucket 信息两种路径。
- [x] 5.5 新增 Explorer 筛选 helper 测试，覆盖 case type、weak modality、horizon、bucket、gain threshold 和排序。
- [x] 5.6 使用 `conda run -n kd_mm_beam pytest tests/test_complementarity_analysis.py tests/test_gradio_complementarity_explorer.py` 验证新增测试。
- [x] 5.7 使用 `conda run -n kd_mm_beam pytest tests/test_conditional_utility_metrics.py tests/test_modality_visual_diagnostics.py` 做相关回归验证。

## 6. 文档与验收

- [x] 6.1 更新 `tools/visualization/README.md` 或相邻文档，说明如何先运行互补分析脚本，再用 `--complementarity-dir` 启动 viewer。
- [x] 6.2 在文档中记录当前 Scene32 输入 schema、weak prediction source、probability metrics 支持范围和输出文件路径。
- [x] 6.3 用一个最小命令示例说明如何筛选 `strong_wrong_weak_correct`、rescue、unused complementary 和 negative transfer。
- [x] 6.4 手动启动 viewer：`conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py --manifest <manifest> --complementarity-dir outputs/scene32/complementarity_analysis --host 127.0.0.1 --port 7860`，确认 Explorer 可加载、筛选、点击样本和导出 CSV。
