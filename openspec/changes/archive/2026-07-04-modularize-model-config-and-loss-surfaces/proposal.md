## Why

模型、配置和 loss 表面已经承载多条 current workflow：`models/modular.py`、`models/jepa.py`、`jepa_downstream.py`、`losses/u_mask_beam_jepa.py` 和 `config/canonical.py` 体量较大。它们需要以 owner 职责为中心拆分，避免模型 forward、配置 recipe、loss extension 和 migration guard 继续互相耦合。

## What Changes

- 将 `ModularSequenceModel` 的 encoder/core/head config normalization、forward stage payload、runtime/auxiliary output assembly 和 geometry/reranker attachment 拆到窄 helper。
- 将 JEPA model/downstream 的 query/token builder、checkpoint reuse、diagnostic output 和 architecture metadata 拆分。
- 将 U-MaskBeamJEPA loss/extension 的 missing-pattern DRO、BTAPA/prototype target、config normalization 和 epoch logging 分离。
- 将 canonical config 的 virtual recipe、path alias、overlay resolution 和 retired-route guard 进一步分层。
- 保持 registry 名称、`model.primary` 构建语义、forward output contract、training extension hook 和 config load 行为兼容。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `modular-sequence-model`: 增加模型 config/forward stage/helper owner 边界要求。
- `model-architecture-extension-contract`: 保持 registry、component baseline、whole-model exception 和 architecture summary 的扩展契约。
- `gps-conditioned-jepa-pretraining`: 固定 JEPA encoder/token/checkpoint reuse 的拆分边界。
- `jepa-downstream-extensibility`: 固定 downstream query/pooling/head/diagnostic 输出兼容要求。
- `u-mask-beam-jepa`: 明确 U-MaskBeamJEPA loss/extension、MP-DRO、BTAPA/prototype target 的 owner 边界。
- `canonical-config-resolution`: 增加 canonical recipe、virtual config、overlay 和 migration guard 的模块化要求。
- `project-import-surface-consolidation`: 防止拆分后新增 package barrel、compat wrapper 或跨领域 `utils` 聚合。

## Impact

- 影响源码：`src/kd_sensing/models/modular.py`、`jepa.py`、`jepa_downstream.py`、`architecture_summary.py`、`losses/u_mask_beam_jepa.py`、`config/canonical.py`、`config/migration_guards.py`。
- 影响测试：`tests/test_modular_sequence_next_query_transformer.py`、`tests/test_gps_conditioned_jepa.py`、`tests/test_u_mask_beam_jepa.py`、`tests/test_config_load_characterization.py`、`tests/test_component_registry.py`、`tests/test_model_architecture_summary.py`。
- 不新增 whole-model exception，不恢复退役 registry 名或旧 config alias。
