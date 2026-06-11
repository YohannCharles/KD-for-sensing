## 1. OpenSpec 和目录审计

- [x] 1.1 记录 cache layout 变更 proposal/design/spec/tasks，并运行 strict validate。
- [x] 1.2 盘点根 `cache/`、`outputs/cache/`、`dataset/*/*cache` 和配置/代码默认路径。

## 2. 默认 cache 路径收敛

- [x] 2.1 在 dataset layout helper 中新增 runtime cache 根和各数据集 cache 默认路径。
- [x] 2.2 将 DeepSense6G image-derived 与 LiDAR BEV cache 的未配置默认路径切到 `outputs/cache/DeepSense6G/<scenario>/...`，保留显式旧路径兼容。
- [x] 2.3 将 MMW image-derived、LiDAR BEV 和 physical label cache 默认路径切到 `outputs/cache/MMW/<condition>/...` 与 `outputs/cache/physical_labels`。
- [x] 2.4 Raymobtime s008 已由 `remove-raymobtime-s008` 退役删除；本 cache layout change 不再定义其默认 cache 路径。

## 3. 配置、文档和本地迁移

- [x] 3.1 更新 preprocess/fusion 配置中的默认 cache 路径，避免新命令默认写入 `dataset/*/*cache` 或根 `cache/`。
- [x] 3.2 更新 README/docs/AGENTS 或相关说明，明确 `dataset/`、`outputs/cache/`、`outputs/`、`logs/` 的角色。
- [x] 3.3 将低风险根目录 `cache/physical_labels` 移动到 `outputs/cache/physical_labels`；不自动移动大型 dataset cache。

## 4. 验证

- [x] 4.1 新增或更新 focused tests，覆盖默认 cache layout、显式旧路径兼容和 physical label 默认值。
- [x] 4.2 运行 `openspec validate consolidate-runtime-cache-layout --strict`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_beam_label_calibration.py -q`。
- [x] 4.4 如配置加载或架构边界受影响，运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_architecture_boundaries.py -q`。
  - 已通过；Raymobtime s008 配置删除由 `remove-raymobtime-s008` 退役 change 处理。
