## 1. CxD 聚合核心

- [x] 1.1 在 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 中拆出 CxD phase analysis 纯函数，覆盖 phase diagram rows、heatmap matrix、RSI、relative drop 和 worst-case 标记。
- [x] 1.2 实现 incomplete grid 检测，对缺失 `(Cx, Dy)` rows 的模型标记 `incomplete_cxd_grid`，不补零、不插值。
- [x] 1.3 保留现有 `results/scenario_d_image_observability.csv` 和 `results/heatmap_cx_dy.npy` 输出兼容性，同时新增 `results/cxd_phase_diagram.csv` 与 `results/cxd_phase_heatmap.npy`。

## 2. Dominance 诊断

- [x] 2.1 定义 dominance row schema，包含 model、group、gps_condition、image_condition、seed、split、三个 contribution score、diagnostic source/status 和 unavailable reason。
- [x] 2.2 实现 gradient norm contribution 计算，按 `gps_norm / (gps_norm + image_norm)` 与同分母 image score 归一化，分母缺失或为零时标记 unavailable。
- [x] 2.3 实现 attention/fusion weights 与 JEPA latent variance 的只读 ingestion，并记录聚合口径。
- [x] 2.4 移除正式 dominance 输出对模型组启发式比例的依赖；启发式只允许用于 smoke 或明确标记为 unavailable/mock 的场景。

## 3. Crossing 与 failure decomposition

- [x] 3.1 实现 strict comparable CNN/AE/ResNet vs JEPA/query-pool crossing detection，输出 `results/crossing_region_Cx_Dy.json`。
- [x] 3.2 实现 low-degradation、JEPA robust regime 和 no-crossing label，并记录 metric margin 与参与配对模型。
- [x] 3.3 实现 GPS-biased JEPA 与 GPS-query-pool JEPA 的 `query_pool_shift` summary。
- [x] 3.4 基于 `(C0,D0)`、`(Cx,D0)`、`(C0,Dy)`、`(Cx,Dy)` 实现 failure-mode decomposition，输出 `results/failure_mode_decomposition.csv`。

## 4. Runner、manifest 与图表产物

- [x] 4.1 扩展 benchmark manifest 解析，支持 `analysis.cxd_phase_transition` 的 enabled flag、paired models、diagnostic sources、fallback policy、thresholds 和 artifact plan。
- [x] 4.2 将新增 artifact 注册到 `benchmark_manifest.json` 的 output registry / `output_files`，包括 generated 与 skipped 状态。
- [x] 4.3 新增或更新图表 writer，输出 `plots/cxd_accuracy_heatmap.png`、`plots/cnn_jepa_crossover_curve.png` 和 `plots/modality_dominance_heatmap.png`，matplotlib 不可用时保留 CSV/JSON/NPY 并记录 warning。
- [x] 4.4 更新 `configs/diagnostics/jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml` 或新增同 family smoke manifest，声明 CxD analysis 配置但保持 mock/smoke caveat。

## 5. 测试

- [x] 5.1 为 CxD phase aggregation、incomplete grid、heatmap shape 和 artifact registration 增加 focused tests。
- [x] 5.2 为 dominance unavailable、gradient norm ingestion、external diagnostics mismatch 和 heuristic fallback 拒绝/降级增加 focused tests。
- [x] 5.3 为 CNN vs JEPA crossing、query_pool shift 和 failure-mode decomposition 增加 synthetic rows tests。
- [x] 5.4 为 CxD difficulty no label shift guard 增加或扩展 synthetic batch tests，验证 target、beam power、soft target、sample id 和 split metadata 不移动。
- [x] 5.5 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_modality_difficulty.py -q`。

## 6. 文档与校验

- [x] 6.1 同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md` 和 `docs/experiment_matrix.md` 中 Scenario D / shortcut benchmark 的 artifact、状态和 caveat。
- [x] 6.2 如 manifest、CLI 或 output files 影响 README 当前 quickstart，更新 README 中 JEPA shortcut benchmark 说明。
- [x] 6.3 运行 `openspec validate add-cxd-phase-transition-analysis --strict`。
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`；若触碰 CLI help 或 pyproject，再运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
