## 1. 边界说明

- [x] 1.1 修正 `src/kd_sensing/baselines/__init__.py` 的 package marker，明确它是 workflow/paper reproduction owner。
- [x] 1.2 更新 `docs/model_architecture_inventory.md`，加入 baseline/model 放置规则表。
- [x] 1.3 更新 `docs/project_surface_inventory.md` 与 `docs/extension_guide.md`，同步源码和配置归属口径。

## 2. 架构护栏

- [x] 2.1 在 `tests/test_architecture_boundaries.py` 增加检查，拒绝 `src/kd_sensing/baselines/` 内 registry 注册。
- [x] 2.2 在 `tests/test_architecture_boundaries.py` 增加检查，拒绝 `src/kd_sensing/models/` 反向导入 `kd_sensing.baselines`。

## 3. 验证

- [x] 3.1 运行 `openspec validate standardize-baseline-model-boundaries --strict`。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
