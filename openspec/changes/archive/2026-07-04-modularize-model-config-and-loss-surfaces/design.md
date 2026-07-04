## Context

模型与配置层承载 current registry、`model.primary`、JEPA、U-MaskBeamJEPA、geometry prior、safe reranker、architecture summary 和 canonical config。当前热点并非都应机械拆分：`ModularSequenceModel` 和 JEPA 模型本身具有领域内聚性，但 config normalization、forward stage payload、checkpoint reuse 和 loss extension logging 已经适合抽出窄 owner。

## Goals / Non-Goals

**Goals:**
- 让模型构建、forward stage、runtime metadata 和 auxiliary outputs 的边界更清楚。
- 拆分 JEPA token/query/checkpoint/diagnostic helper。
- 拆分 U-MaskBeamJEPA loss extension 的 MP-DRO、BTAPA/prototype、logging 和 config normalization。
- 拆分 canonical config recipe、virtual alias 和 retired-route guard。

**Non-Goals:**
- 不新增 whole-model exception。
- 不恢复退役 registry 名、KD alias 或旧 facade。
- 不改变 forward output、adapt_model_output、training extension hook 或 config load 语义。

## Decisions

1. **模型 forward 采用 stage payload helper。**
   保留 `ModularSequenceModel.forward` public 行为，内部用 typed/dict payload 在 encoder、core、head、post-process 和 runtime attachment 间传递。

2. **配置 recipe 与 migration guard 分层。**
   canonical recipe/overlay/path alias 与 retired route guard 分离，避免新增 current config 时触碰退役拒绝逻辑。

3. **loss extension 不变更训练 hook。**
   先拆 MP-DRO 和 BTAPA/prototype helper，不改变 extension context、after-forward hook 或 epoch metadata。

4. **architecture summary 只读。**
   重构不得让 summary helper参与模型构建或注册，只消费 resolved config 与 nn.Module。

## Risks / Trade-offs

- 模型 forward 行为漂移 -> 使用 synthetic forward、metadata 和 architecture summary tests 覆盖。
- config alias 意外接管退役路径 -> 保持 migration guard focused tests。
- loss extension log 字段变化 -> tests 固定 epoch log、metadata 和 missing-pattern outputs。

## Migration Plan

1. 捕获 registry build、forward output 和 config load focused baseline。
2. 拆 config/forward 纯 helper，保持 import owner 不变。
3. 拆 JEPA checkpoint/query/diagnostic helper。
4. 拆 U-MaskBeamJEPA loss extension helper。
5. 运行 `openspec validate modularize-model-config-and-loss-surfaces --strict`、`conda run -n kd_mm_beam pytest tests/test_modular_sequence_next_query_transformer.py tests/test_gps_conditioned_jepa.py tests/test_u_mask_beam_jepa.py tests/test_config_load_characterization.py tests/test_component_registry.py tests/test_model_architecture_summary.py tests/test_architecture_boundaries.py -q`。

## Open Questions

- `ModularSequenceModel` stage payload 使用 dataclass 还是内部 dict 更合适？
- canonical recipe 是否需要独立子模块，还是在 `canonical.py` 中保持公共入口并移动私有表？
