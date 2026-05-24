## 1. Inventory And Guardrails

- [x] 1.1 更新或新增热点模块 inventory，列出 objective metadata、Multimodal-NF helper、viewer manifest、DeepVerse label builder 的当前职责和目标拆分。
- [x] 1.2 标记保留为公开兼容 facade 的模块，以及内部代码不得新增引用的二级聚合路径。
- [x] 1.3 扩展架构边界测试，拒绝内部代码对新增兼容 facade 的回流依赖。

## 2. Objective Metadata Decomposition

- [x] 2.1 将 objective 名称、默认指标、metric mode 和 alias 表拆到窄模块或子包。
- [x] 2.2 将 history fields 与 TensorBoard scalar schema 拆到窄模块。
- [x] 2.3 保持原 `kd_sensing.engine.objective_metadata` 公开 API 兼容，并改为调用窄模块。
- [x] 2.4 使用 `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_training_io_workflow.py -q` 中相关用例验证行为兼容。

## 3. Multimodal-NF Helper Decomposition

- [x] 3.1 将 Multimodal-NF path resolution 与 layout helper 从 common helper 中拆出。
- [x] 3.2 将 audit、HDF5 inspection 和 codebook metadata helper 拆到窄模块。
- [x] 3.3 将 index row 构建、split assignment 和 index writer/loader 拆到窄模块。
- [x] 3.4 保持 preprocessor registry 名称和 `multimodal_nf_common` 公开导出兼容。
- [x] 3.5 使用 `conda run -n kd_mm_beam pytest tests/test_multimodal_nf_dataset.py -q` 验证。

## 4. Viewer Manifest Decomposition

- [x] 4.1 将 viewer manifest sample schema、cache metadata、path resolution 和 writer 拆到窄模块。
- [x] 4.2 将 predictions/quality/gate merge 逻辑拆到独立 helper。
- [x] 4.3 保持 `kd-sensing-export-viewer-manifest`、Gradio viewer 和相关公开 import 行为兼容。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest tests/test_gradio_complementarity_explorer.py tests/test_modality_visual_diagnostics.py -q` 或新增 focused tests 验证。

## 5. DeepVerse Label Builder Decomposition

- [x] 5.1 拆分 scene metadata/config-m resolution helper。
- [x] 5.2 拆分 target derivation、split assignment、sanity check 和 output writer。
- [x] 5.3 保持 `scripts/deepverse/generate_dt31_cache.py` 行为和配置入口兼容。
- [x] 5.4 使用 `conda run -n kd_mm_beam pytest tests/test_deepverse_dt31_generation.py -q` 验证。

## 6. Artifact Compatibility And Final Checks

- [x] 6.1 为拆分涉及的 `final_config.yaml`、`train_log.json`、`metrics.json` 或 viewer manifest 关键字段补充 presence/compatibility 断言。
- [x] 6.2 更新 docs/project_surface_inventory.md 或等价 inventory 文档。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 6.4 运行 `openspec validate decompose-hotspot-modules --strict`。
- [x] 6.5 根据实际 touched 模块运行相关 focused tests，必要时再运行 `conda run -n kd_mm_beam pytest -q`。
