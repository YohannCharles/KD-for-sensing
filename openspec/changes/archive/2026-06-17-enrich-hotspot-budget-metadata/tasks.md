## 1. Schema 设计

- [x] 1.1 在 `docs/maintainer_context_index.yaml` 中声明 hotspot priority/status 允许值。
- [x] 1.2 为 `symbol_budgets` 和 `file_budgets` 增加 priority、status、split_targets、rationale、validation_commands 和可选 next_change。
- [x] 1.3 确保 JEPA benchmark、DeepSense6GDataset、trainer、BeamBench workflow 等现有 budget 都有行动元数据。

## 2. 校验接入

- [x] 2.1 更新 maintainer context index validator，检查 hotspot action metadata。
- [x] 2.2 保持现有行数 budget 检查语义不变。
- [x] 2.3 检查 validation commands 使用 `conda run -n kd_mm_beam`。

## 3. 文档同步

- [x] 3.1 更新 `docs/project_surface_inventory.md`，说明 index 中新增 action metadata 的职责。
- [x] 3.2 确认 inventory 仍保留热点原因和暂缓说明。

## 4. 验证

- [x] 4.1 运行 `openspec validate enrich-hotspot-budget-metadata --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
