## Why

Multimodal-NF 已经支持审计、index、单模态/fusion 配置、near-field beam selection、LOS 和 selection multitask，但部分 runtime 命名仍偏向早期 future beam 语义，且 archived spec 中仍保留 TBD Purpose。随着当前实验矩阵扩大，运行产物需要更精确地表达 objective、target schema、codebook、split 和 task semantics，避免后续横向比较和复现实验时误读。

本变更整理 Multimodal-NF 的运行语义和文档契约，不改变核心训练算法，重点让 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json` 和 runtime metadata 更可解释。

## What Changes

- 补齐 `multimodal-nf-dataset` 和 `dataset-runtime-contracts` 的真实 Purpose，移除 archived TBD 文案。
- 明确 Multimodal-NF 支持的 objective 语义：`near_field_beam_selection`、`current_los_classification`、`current_link_quality` 和 `selection_multitask`。
- 调整 runtime metadata 命名，使 `prediction_setup.task_semantics`、`target_schema`、`objective`、`codebook_shape`、`input_profiles` 和 split metadata 与当前 objective 对齐。
- 明确 `near_field_beam_selection` 与 Raymobtime `current_beam_selection` 的差异：前者使用 Multimodal-NF 近场三维 codebook flatten label，后者使用 Raymobtime 当前 snapshot beam pair class。
- 增加运行产物一致性检查，确保 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json`、`metrics.json` 中的 dataset family、objective、modalities、num_classes 和 codebook metadata 不互相矛盾。
- 更新 README/docs 中 Multimodal-NF 实验矩阵说明，区分 dataset smoke、单任务、multitask 和 fusion 配置入口。

## Capabilities

### New Capabilities
- 无。

### Modified Capabilities
- `multimodal-nf-dataset`: 补齐 capability 目的说明，并明确当前支持的 target/objective/runtime metadata 语义。
- `dataset-runtime-contracts`: 补齐 capability 目的说明，并加强 runtime metadata 与 target schema 的一致性要求。
- `first-class-prediction-tasks`: 明确 `near_field_beam_selection` 与 selection multitask 在 Multimodal-NF 下的指标、target schema 和 runtime metadata 要求。
- `experiment-workflow`: 增加运行产物一致性检查和 Multimodal-NF 配置/产物可解释性要求。

## Impact

- 主要影响 `src/kd_sensing/engine/run_metadata.py`、`src/kd_sensing/engine/objective_metadata.py`、Multimodal-NF dataset/preprocessing metadata helper、配置校验和文档。
- 不改变默认训练入口、模型结构、loss 数学定义或已生成本地 outputs。
- 测试覆盖 spec TBD 清理、runtime metadata 字段、Multimodal-NF objective 产物一致性和相关配置加载。
