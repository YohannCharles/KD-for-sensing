## 1. Final analysis scripts

- [x] 1.1 新增 `scripts/significance_tests.py`，输出 significance summary、bootstrap deltas、per-scene deltas 和 paper significance table。
- [x] 1.2 新增 `scripts/export_pattern_heatmap.py`，输出 pattern metric matrix、delta、win-count summary 和论文图。
- [x] 1.3 完善 `scripts/profile_scenes31_34_methods.py`，补齐 family、table output 和 inference-cost notes。
- [x] 1.4 新增 `scripts/plot_error_cdf.py`，输出 all-missing、miss3、missing-ratio-75 的 CDF 数据和图。
- [x] 1.5 新增 `scripts/summarize_sampling_distribution.py`，输出 missing-count/subset 分布、summary 和图。

## 2. Final tables and conclusion

- [x] 2.1 新增 `scripts/update_final_paper_tables.py`，整合 summary、statistics、pattern、profile、CDF 和 sampling artifacts。
- [x] 2.2 扩展 `scripts/write_scenes31_34_main_conclusion.py`，读取 final analysis artifacts 并按八项证据写保守结论。
- [x] 2.3 新增 `scripts/run_final_scene31_34_analysis.sh`，按顺序运行全部 final analysis。

## 3. OpenSpec and tests

- [x] 3.1 更新 OpenSpec delta，记录 significance、pattern heatmap、compute profile、error CDF、sampling distribution 和 final paper table update。
- [x] 3.2 新增 `tests/test_scene31_34_final_analysis.py`，覆盖 summary fallback、heatmap、sampling、CDF、final notes 和 mask_suspect exclusion。
- [x] 3.3 运行 `openspec validate promote-scenes31-34-main-missing-count --strict`。
- [x] 3.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。

## 4. Final polish audit

- [x] 4.1 审计 `significance_summary.csv`，拆分 seed mean delta 与 bootstrap mean/CI，并补 fraction/pp/beam-index 单位字段和 sanity notes。
- [x] 4.2 为 `profile_scenes31_34_methods.py` 增加 `--benchmark-latency`、GPU/device、warmup/benchmark batch 和 failure-to-NaN 行为。
- [x] 4.3 生成论文友好 degradation curves、pattern delta/grouped heatmap 和 error CDF paper variants。
- [x] 4.4 更新 final paper table notes 与 final conclusion 的保守主结论。
- [x] 4.5 新增 `scripts/run_final_scene31_34_polish.sh` 一键 final polish runner。
