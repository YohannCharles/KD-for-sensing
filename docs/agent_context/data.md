# 数据与 batch 任务上下文

用于 dataset、batch contract、modality profile、target provider、split、数据准备和本地数据边界相关改动。不要读取或提交真实 `dataset/` 内容，除非用户明确要求执行本地数据任务。

## 先读

- `openspec/specs/dataset-directory-layout/spec.md`
- `openspec/specs/dataset-runtime-contracts/spec.md`
- `openspec/specs/modality-contracts/spec.md`
- `openspec/specs/dataset-loader-behavior/spec.md`
- `docs/project_surface_inventory.md` 中 data/training runtime wave 和本地产物边界

## Owner

- DeepSense6G：`src/kd_sensing/data/datasets/deepsense6g.py` 与 `src/kd_sensing/data/datasets/deepsense6g_*`
- MMW family：`src/kd_sensing/data/datasets/mmw_family_adapter.py`、`src/kd_sensing/data/mmw/`
- Batch contract：`src/kd_sensing/engine/batch.py`、`src/kd_sensing/engine/batch_targets.py`
- Modality registry/profile：`src/kd_sensing/modalities.py`、`src/kd_sensing/data/difficulty/`

## 边界

- `dataset/` 是本地输入；源码只保留 `dataset/.gitkeep`。
- 新 cache 默认写入 `outputs/cache/` 或明确 ignored runtime root。
- dataset tests 应使用 synthetic rows、tmp path 或 fixture，不依赖用户本地数据。
- 新增 target、label adapter、history anchor 或 sensitive field guard 时，优先落到窄 helper，不把规则塞回大 dataset owner。

## 验证

- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- DeepSense6G contract: `conda run -n kd_mm_beam pytest tests/test_deepsense6g_contract_helpers.py -q`
- Batch/objective: `conda run -n kd_mm_beam pytest tests/test_prediction_objectives.py tests/test_evaluation_pass.py -q`
- MMW preparation: `conda run -n kd_mm_beam pytest tests/test_mmw_town10_preparation.py -q`
