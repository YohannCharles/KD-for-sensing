## Why

DeepSense6G/MMW dataset、训练 runtime、batch contract 和 evaluation pass 是当前训练链路的高维护成本区域。它们影响面大，不适合和诊断或脚本清理混在一起，需要单独 change 分 wave 做行为保持重构。

## What Changes

- 继续拆分 `DeepSense6GDataset` 中的 label/history adapter、resource reader glue、scaler/normalizer setup、target provider setup 和 cache path 规则。
- 收敛 `MMWDataset` 与 `MMWFamilyAdapter` 的边界，分离 geometry、physical label、beam power、radio/path semantic 和 physics supervision。
- 将训练 runtime 的 context 准备、resource build、state restore、epoch loop、checkpoint/finalization 进一步阶段化，但不改变训练数学语义。
- 拆分 `engine.batch` 和 `engine.evaluation_pass` 中的 modality target preparation、label adapters、objective outputs、metric aggregation 和 prediction metadata。
- 拆分 MMW GPS v2 workflow 的 label-space resolution、support selection、protocol summary 和 artifact writer。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `dataset-runtime-contracts`: 明确 dataset contract helper、family adapter 和 target provider 的 owner 边界。
- `dataset-loader-behavior`: 保持 DataLoader、cache、按需加载和 split 行为兼容，并要求 focused tests。
- `training-evaluation-runtime`: 固定训练 context/loop/finalization 和 evaluation pass 的阶段边界。
- `first-class-prediction-tasks`: 保持 objective metadata、prediction target、history fields 和 metric schema 兼容。
- `mmw-sensor-assisted-beam-prediction`: 明确 MMWDataset family adapter、physical label 和 beam power owner 边界。
- `mmw-town-gps-adapter-v2`: 明确 MMW GPS v2 workflow 的 protocol/artifact writer 拆分要求。
- `project-hotspot-governance`: 更新 dataset/training/evaluation/MMW wave 的热点预算和验证边界。

## Impact

- 影响源码：`src/kd_sensing/data/datasets/deepsense6g.py`、`mmw.py`、`mmw_family_adapter.py`、`engine/trainer.py`、`engine/batch.py`、`engine/evaluation_pass.py`、`engine/mmw_town_gps_v2.py`。
- 影响测试：`tests/test_training_io_workflow.py`、`tests/test_deepsense6g_contract_helpers.py`、`tests/test_mmw_town10_preparation.py`、`tests/test_evaluation_pass.py`、`tests/test_prediction_objectives.py`、`tests/test_architecture_boundaries.py`。
- 不改变数据 split、beam label 口径、soft label、checkpoint schema、run metadata 或默认输出分区。
