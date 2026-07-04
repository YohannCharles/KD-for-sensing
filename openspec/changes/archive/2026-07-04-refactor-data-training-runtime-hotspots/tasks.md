## 1. Baseline 与顺序

- [x] 1.1 记录 `DeepSense6GDataset`、`MMWDataset`、`engine.batch`、`evaluation_pass`、`trainer` 和 `mmw_town_gps_v2` 的当前职责与 focused tests。
- [x] 1.2 更新 `docs/project_surface_inventory.md` 中 data/training runtime wave 的热点规模、拆分方向和验证命令。

## 2. Dataset wave

- [x] 2.1 将 DeepSense6G label/history adapter、resource reader glue、target provider setup、scaler/normalizer setup 中的纯规则迁入窄 helper。
- [x] 2.2 将新增或现有 GPS feature mode、beam target source、column guard、cache path rule 固定在 contract/cache/target helper。
- [x] 2.3 收敛 MMW family adapter 中 geometry、availability、radio/path semantic、physical label、beam power 和 physics supervision 边界。
- [x] 2.4 补充 synthetic dataset tests，不读取真实 `dataset/`。

## 3. Runtime wave

- [x] 3.1 拆分 training context preparation、resource build、restore、loop/finalize helper，保持 `_train_inner` 薄入口。
- [x] 3.2 拆分 `engine.batch` 的 modality target preparation、label adapters 和 history anchor input helper。
- [x] 3.3 拆分 `run_evaluation_pass` 的 objective output、metric aggregation、prediction metadata 和 difficulty stage helper。
- [x] 3.4 拆分 MMW GPS v2 label-space resolution、support selection、protocol summary 和 artifact writer。

## 4. 验证

- [x] 4.1 运行 `openspec validate refactor-data-training-runtime-hotspots --strict`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_deepsense6g_contract_helpers.py tests/test_mmw_town10_preparation.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_evaluation_pass.py tests/test_prediction_objectives.py tests/test_architecture_boundaries.py -q`。
- [x] 4.4 高风险 runtime wave 完成后运行 `conda run -n kd_mm_beam pytest -q`，无法运行时记录原因和剩余风险。
