## Why

JEPA visual analysis 和 JEPA GPS shortcut benchmark 已成为当前诊断主线，但实现文件继续膨胀：`jepa_visual_analysis.py` 约 3416 行，`jepa_benchmark_runner.py` 约 2370 行。它们适合做行为保持的内部模块化，降低新增图表、suite、artifact schema 时的回归风险。

## What Changes

- 将 JEPA visual analysis 拆分为配置/schema、模型分析 loop、表格输出、图像输出、case payload、benchmark/evidence 集成、report/manifest builder 等窄 owner。
- 将 JEPA benchmark runner 拆分为 suite dispatch、model metric source、predictive artifact planning、claim gate、real-forward diagnostics、output registry 和 manifest writer。
- 保持 `kd-sensing-jepa-visual-analysis`、`kd-sensing-jepa-gps-shortcut-benchmark`、公开 facade 和现有 output schema 兼容。
- 防止 suite-specific helper 回流到 `jepa_gps_shortcut_benchmark.py` 公开 facade。
- 不改变 benchmark comparability、Predictive Robustness、Scenario C/D/CxD、GPS-query evidence 或 JEPA visual report 的指标语义。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `jepa-visual-analysis-suite`: 增加内部 owner 边界、artifact/report writer 兼容和 focused validation 要求。
- `jepa-gps-shortcut-benchmark`: 增加 benchmark runner 模块化边界、suite dispatch 和 output schema 兼容要求。
- `predictive-jepa-robustness`: 保持 predictive summary、claim gate 和 diagnostics bundle 的字段兼容，同时允许内部拆分。
- `real-perturbation-forward-evaluation`: 固定 real-forward shard/cache/diagnostics 的模块边界和验证要求。
- `project-hotspot-governance`: 更新 JEPA diagnostics 热点预算、拆分方向和 facade 回流防护。
- `project-import-surface-consolidation`: 继续禁止内部代码从公开 facade 导入 private helper。

## Impact

- 影响源码：`src/kd_sensing/diagnostics/jepa_visual_analysis.py`、`jepa_benchmark_runner.py`、`jepa_benchmark_*`、`gps_query_evidence.py`。
- 影响 CLI：`src/kd_sensing/cli/jepa_visual_analysis.py`、`src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py` 应保持薄 glue。
- 影响测试：`tests/test_jepa_visual_analysis.py`、`tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_modality_difficulty.py`、`tests/test_architecture_boundaries.py`。
- 默认输出仍在 ignored `outputs/visual_analysis/`、`outputs/analysis/` 或用户显式路径。
