## 1. Baseline 捕获

- [x] 1.1 记录 current registry 名称、removed-name guard、`model.primary` config load、synthetic forward 和 architecture summary focused tests。
- [x] 1.2 更新 `docs/project_surface_inventory.md` 中模型/config/loss 热点的 accepted/split-next 说明。

## 2. 模型与 JEPA 拆分

- [x] 2.1 抽出 ModularSequenceModel encoder/projector/core/head config normalization helper。
- [x] 2.2 抽出 ModularSequenceModel forward stage payload、runtime output、auxiliary output、geometry prior/reranker attachment helper。
- [x] 2.3 抽出 JEPA tokenizer/query/checkpoint reuse/diagnostic output helper，保持 metadata 兼容。
- [x] 2.4 抽出 JEPA downstream query/pooling/head/diagnostic metadata helper。

## 3. Loss 与配置拆分

- [x] 3.1 拆分 U-MaskBeamJEPA MP-DRO、BTAPA/prototype target、loss config normalization、epoch logging helper。
- [x] 3.2 拆分 canonical config virtual recipe、overlay/path alias 和 retired-route migration guard。
- [x] 3.3 检查拆分后没有新增 package barrel、compat wrapper、旧 registry alias 或跨领域 `utils` 聚合。

## 4. 验证

- [x] 4.1 运行 `openspec validate modularize-model-config-and-loss-surfaces --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_gps_conditioned_jepa.py tests/test_u_mask_beam_jepa.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_model_architecture_summary.py tests/test_architecture_boundaries.py -q`。
