## 1. Phase 1.5 决策语义

- [x] 1.1 调整 `build_phase_1_5_summary()` 的 final gate，要求 bootstrap、checkpoint matrix 和 baseline matrix 均为 `complete` 后才允许总 `decision.status=complete`。
- [x] 1.2 在 checkpoint matrix 未完成时保留 bootstrap / baseline 局部结果，但总 `decision.label=pending` 且 `decision.evidence_level=exploratory`。
- [x] 1.3 扩展 `tests/test_phase_1_5_utility_validation.py`，覆盖 checkpoint missing、audit pending、baseline pending 与全部完成四种状态。
- [x] 1.4 使用 `conda run -n kd_mm_beam pytest tests/test_phase_1_5_utility_validation.py -q` 验证 Phase 1.5 回归。

## 2. 轻量导入边界

- [x] 2.1 将 `kd_sensing.engine.__init__` 改为 lazy export，避免导入轻量子模块时 eager 加载 builders、trainer、evaluator 或 validator。
- [x] 2.2 将 `kd_sensing.diagnostics.__init__` 改为 lazy export，避免导入 `g2d_diagnostics` 等轻量子模块时 eager 加载 visualization core、viewer manifest 或 matplotlib。
- [x] 2.3 将 `kd_sensing.distillation.__init__` 改为 lazy export，避免导入 `g2d_smp` 等工具子模块时 eager 加载 distillers、engine builders 或 dataset transforms。
- [x] 2.4 新增或扩展架构边界测试，使用独立 Python 子进程检查 `sys.modules`，验证轻量导入不加载 `_builders_impl`、`_legacy`、`pandas`、`scipy`、`matplotlib`。
- [x] 2.5 使用 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q` 验证导入边界。

## 3. Engine builder 真实拆分

- [x] 3.1 将 cache policy 实现迁移到 `src/kd_sensing/engine/cache_policy.py`，保留旧导入兼容。
- [x] 3.2 将 dataset/dataloader 构建实现迁移到 `src/kd_sensing/engine/data_factory.py`，保留 `build_dataloaders()` 等行为不变。
- [x] 3.3 将 normalization artifact 读写实现迁移到 `src/kd_sensing/engine/normalization_artifacts.py`。
- [x] 3.4 将 run metadata / throughput / cache metadata 实现迁移到 `src/kd_sensing/engine/run_metadata.py`。
- [x] 3.5 将 model / loss / distiller / optimizer / scheduler / device 构建实现迁移到 `src/kd_sensing/engine/optim.py`。
- [x] 3.6 将 `src/kd_sensing/engine/builders.py` 保持为兼容 facade，并减少或清空 `_builders_impl.py` 中的主实现。
- [x] 3.7 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_student_configs.py tests/test_architecture_boundaries.py -q` 验证 builder 拆分。

## 4. 数据转换 legacy 拆分

- [x] 4.1 将 image motion mask、cache key 和 image IO 实现迁移到 `data/transform_ops/image.py` 或通用 IO/cache 模块。
- [x] 4.2 将 GPS feature、GPS scaler 和 GPS artifact 加载实现迁移到 `data/transform_ops/gps.py` 或 normalization 模块。
- [x] 4.3 将 LiDAR BEV、LiDAR cache、LiDAR normalizer 和 streaming stats 实现迁移到 `data/transform_ops/lidar.py`。
- [x] 4.4 将 mmWave power vector 和 scaler 实现迁移到 `data/transform_ops/mmwave.py`。
- [x] 4.5 将 radar map 读取和通用 `joined_resource` / atomic save 等实现迁移到对应 radar、IO 或 cache 模块。
- [x] 4.6 保留 `data.transform_ops._legacy` 和 `data.transforms` 的旧公共符号兼容，并补充 import compatibility 测试。
- [x] 4.7 使用 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_preprocessing_formats.py tests/test_gps_modality.py tests/test_lidar_modality.py tests/test_mmwave_modality.py -q` 验证转换拆分。

## 5. CLI entry points 与文档

- [x] 5.1 校正 `pyproject.toml` 中 `kd-sensing-export-viewer-manifest` 的目标函数，确保它指向 manifest export CLI 而非不完整兼容入口。
- [x] 5.2 确认 `kd-sensing-visualize-modalities` 的帮助信息与当前 Gradio / manifest 工作流一致。
- [x] 5.3 在 `kd_mm_beam` 中执行 `conda run -n kd_mm_beam python -m pip install -e .` 刷新 editable install 元数据。
- [x] 5.4 使用 `conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help` 和 `conda run -n kd_mm_beam kd-sensing-visualize-modalities --help` 验证 entry points。
- [x] 5.5 更新 README 或 `tools/visualization/README.md`，记录推荐 CLI、`python tools/...` fallback 和入口验证方式。

## 6. 产物边界与健康检查

- [x] 6.1 在 README 或扩展指南中明确 `All_models/*.pth` 是内置复现权重还是待外部化资产，并说明新生成 checkpoint 不应提交。
- [x] 6.2 增加快速健康检查文档或脚本，覆盖轻量导入 smoke、entry point help、Phase 1.5 pending gate 和互补分析核心测试。
- [x] 6.3 使用 `conda run -n kd_mm_beam pytest tests/test_complementarity_analysis.py tests/test_gradio_complementarity_explorer.py -q` 验证互补分析未被导入和拆分改动破坏。
- [x] 6.4 使用 `conda run -n kd_mm_beam pytest -q` 执行最终全量回归，要求全部通过。
- [x] 6.5 使用 `openspec validate --all` 和 `openspec status --change stabilize-project-health-and-validation` 验证 OpenSpec 状态。
