## Why

`src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 已成为当前 P0 维护热点：同一文件同时承担 manifest normalization、Scenario C/D/CxD、Predictive Robustness、metric aggregation、artifact writer 和 plotting，接近维护预算上限。继续在单文件中扩展会让 Codex 和维护者更难判断应该改哪一层，也更容易意外改变 CSV/JSON schema。

## What Changes

- 将 JEPA GPS shortcut benchmark runner 拆成职责明确的窄模块，原 `jepa_gps_shortcut_benchmark.py` 保留为公开 facade 和兼容导出。
- 拆分 manifest/schema normalization、Scenario C async GPS、Scenario D image observability、CxD phase analysis、Predictive JEPA robustness、artifact/CSV/JSON writer 和 plotting helper。
- 保持现有 CLI、manifest 输入、输出文件名、CSV/JSON 字段、metric 聚合语义和 visual-analysis ingestion bundle 兼容。
- 更新热点 inventory 和 maintainer context index，使 `jepa_gps_shortcut_benchmark.py` 的文件预算降低，并为新窄模块设置合理预算。
- 不新增公开训练/评估 CLI，不读取真实 `dataset/`，不写入源码内产物，不改变 benchmark 数值语义。

## Capabilities

### New Capabilities

### Modified Capabilities

- `jepa-gps-shortcut-benchmark`: benchmark runner 内部结构拆分，要求公开契约和输出 schema 保持兼容。
- `project-health-guardrails`: 热点预算和 facade 回流检查需要覆盖拆分后的 JEPA benchmark 窄模块。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py` 及新增 `src/kd_sensing/diagnostics/jepa_benchmark_*.py` 或等价窄模块。
- 影响 `tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_architecture_boundaries.py`、`docs/maintainer_context_index.yaml` 和 `docs/project_surface_inventory.md`。
- 需要运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py tests/test_architecture_boundaries.py -q`。
