## 1. Baseline 捕获

- [x] 1.1 为 JEPA visual analysis smoke fixture 捕获 manifest、report、table、figure registry 和 skipped-output 关键字段。
- [x] 1.2 为 JEPA GPS shortcut benchmark smoke/predictive/Scenario D fixture 捕获 CSV/JSON/manifest 关键字段。
- [x] 1.3 更新 `docs/project_surface_inventory.md` 中 JEPA diagnostics 热点规模、owner 模块和 facade budget。

## 2. JEPA visual analysis 拆分

- [x] 2.1 抽出 analysis config/schema validation helper，保持 override 和 digest 行为兼容。
- [x] 2.2 抽出 model analysis loop 与 optional model failure handling。
- [x] 2.3 抽出 table writer、figure writer、case payload writer、benchmark/evidence integration。
- [x] 2.4 抽出 report builder 和 `analysis_manifest.json` builder，保持 output registry 兼容。

## 3. JEPA benchmark runner 拆分

- [x] 3.1 抽出 manifest/comparability/model metric source ingestion helper。
- [x] 3.2 抽出 suite dispatch 和 aggregation helper，覆盖 legacy、Scenario C/D/CxD、Predictive 和 GPS-query advantage。
- [x] 3.3 抽出 predictive artifact planning、claim gate 和 diagnostics bundle writer。
- [x] 3.4 抽出 real-forward diagnostics execution/cache/shard helper。
- [x] 3.5 保持 `jepa_gps_shortcut_benchmark.py` 为公开 facade，禁止 suite-specific helper 回流。

## 4. 验证

- [x] 4.1 运行 `openspec validate modularize-jepa-diagnostics-suites --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py tests/test_jepa_gps_shortcut_benchmark.py tests/test_modality_difficulty.py tests/test_architecture_boundaries.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help` 和 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`。
