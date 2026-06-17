## 1. 模块边界

- [x] 1.1 梳理 `jepa_gps_shortcut_benchmark.py` 的公开导出、CLI 依赖和测试直接引用。
- [x] 1.2 新增 manifest/schema、Scenario C、Scenario D/CxD、Predictive、artifact writer 和 plotting 窄模块。
- [x] 1.3 保留原模块 facade，确保公开 import 和 CLI target 不变。

## 2. 行为保持

- [x] 2.1 先迁移纯 normalization、aggregation、CSV/JSON scalar helper，并运行 focused tests。
- [x] 2.2 迁移 Scenario C async GPS helper，保持 source index、mask、fallback 和 warnings schema。
- [x] 2.3 迁移 Scenario D/CxD helper，保持 result CSV、heatmap artifact 和 phase/dominance/crossing schema。
- [x] 2.4 迁移 Predictive JEPA P0-P5 helper，保持 mock/smoke gating 和 strict comparability 字段。
- [x] 2.5 迁移 artifact writer 和 plotting helper，保持输出路径和 graceful fallback。

## 3. 治理与文档

- [x] 3.1 更新 `docs/maintainer_context_index.yaml` 的 hotspot budgets，降低 facade budget 并登记新增窄模块。
- [x] 3.2 更新 `docs/project_surface_inventory.md`，记录拆分后的职责模块和防回流规则。
- [x] 3.3 更新架构边界测试，拒绝 suite-specific helper 回流到 facade。

## 4. 验证

- [x] 4.1 运行 `openspec validate split-jepa-shortcut-benchmark-runner --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.4 如触碰 visual-analysis ingestion，追加运行对应 JEPA visual analysis focused tests。
