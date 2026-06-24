## 1. 基线扫描与范围确认

- [x] 1.1 扫描 tracked imports，列出内部依赖 package barrel、兼容 facade、`config/source.py`、`data/transform_ops/normalization.py` 和重复 helper 的调用点。
- [x] 1.2 确认候选删除项是否有 README、docs、pyproject、current spec 或测试声明的 public import 契约；有契约的只收缩实现，不直接删除路径。
- [x] 1.3 记录不处理项和原因，例如训练语义风险、仍有 current public 契约或 helper 语义不一致。

## 2. 聚合面与包装层收敛

- [x] 2.1 将仓库内部从 package barrel 导入的代码迁移到 owner module 路径。
- [x] 2.2 删除或缩短没有 current public 契约的 eager re-export `__init__.py`，确保轻量导入不拉入 dataset、model、diagnostics、checkpoint registry 或重可视化依赖。
- [x] 2.3 合并 `config/source.py` 的单用途包装，保留 `_base_`、virtual config 和 removed config guard 行为。
- [x] 2.4 移除 `data/transform_ops/normalization.py` 这类单用途 re-export，调用方改为从 `gps.py`、`lidar.py`、`mmwave.py` 等 owner module 导入。

## 3. Registry 与 helper 收敛

- [x] 3.1 梳理 `registries.py` removed-name 表，保留仍有当前迁移价值的名称和错误提示。
- [x] 3.2 移除完全退役且已有 migration guard、retired-tombstone spec 或文档边界覆盖的 removed-name 表项，确认 unknown-name 错误仍包含 registry 和可用名称上下文。
- [x] 3.3 合并语义一致的 `_json_ready`、`_write_csv`、`_read_csv`、`_float_or_none`、slug/path 小 helper；只在确有多处 current 调用且 owner 清晰时新增领域窄 helper。
- [x] 3.4 若某个重复 helper 语义不一致，保留局部私有实现并在最终说明中记录不合并原因。

## 4. 训练 extension 影响评估

- [x] 4.1 用 CodeGraph 或调用点扫描确认 `TrainingExtension`、`JepaTrainingExtension`、`TeacherGuidanceTrainingExtension`、`NoOpTrainingExtension` 的实际影响面。
- [x] 4.2 若删除 extension 框架能保持 JEPA、teacher guidance、checkpoint_loads、diagnostics 和 batch runner 行为不变，则将其直接内联到训练 owner；否则保留并登记为暂缓项。
- [x] 4.3 为执行的路径运行 focused tests，至少覆盖 prediction objective、JEPA 训练 smoke 或 teacher guidance 相关测试。

## 5. 护栏与文档更新

- [x] 5.1 更新 `tests/test_architecture_boundaries.py` 或等价健康检查，拒绝 tracked runtime artifacts、已删除 facade 回流和无契约重依赖 barrel。
- [x] 5.2 增加或调整检查，确保本地 ignored `__pycache__`、`.pyc`、`outputs/`、`logs/` 不会驱动常规架构边界测试失败。
- [x] 5.3 更新 `docs/project_surface_inventory.md`，记录已删除项、保留项、merge-candidate 或 right-size-accepted 理由。
- [x] 5.4 如删除 public import 路径，更新 README、docs 或 OpenSpec 文本，避免继续推荐已删除路径。

## 6. 验证

- [x] 6.1 运行 `openspec validate prune-audit-followup-overengineering --strict`。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_component_registry.py -q`。
- [x] 6.4 如触碰训练 extension 或 objective 路径，运行 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_gps_conditioned_jepa.py -q` 或更窄的相关测试。
- [x] 6.5 如触碰 CLI、README 推荐入口或 pyproject console scripts，运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`。
- [x] 6.6 检查 `git status --short`，确认没有把 `dataset/`、`outputs/`、`logs/`、cache、checkpoint、`.pyc` 或其它本地产物纳入源码变更。
