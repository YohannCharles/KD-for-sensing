## 1. Spec And Documentation Cleanup

- [x] 1.1 补齐 `openspec/specs/multimodal-nf-dataset/spec.md` 的 Purpose，移除 archived TBD 文案。
- [x] 1.2 补齐 `openspec/specs/dataset-runtime-contracts/spec.md` 的 Purpose，移除 archived TBD 文案。
- [x] 1.3 更新 README 或 docs 中 Multimodal-NF 小节，区分 dataset smoke、单任务、multitask 和 fusion 配置入口。
- [x] 1.4 增加文档或测试检查，防止新增 `TBD - created by archiving` Purpose 回流。

## 2. Runtime Metadata Semantics

- [x] 2.1 调整 Multimodal-NF `prediction_setup` metadata，使 objective、task semantics 和 target schema 按当前 objective 输出。
- [x] 2.2 为 `near_field_beam_selection` 输出 codebook shape、flatten order、num beam classes 和 near-field target schema。
- [x] 2.3 为 `current_los_classification` 和 `current_link_quality` 输出对应 target schema，避免误标为 beam-only run。
- [x] 2.4 为 `selection_multitask` 输出 beam、LOS、link 三类 target、head/output 字段、loss 字段和 metric 字段。
- [x] 2.5 保留已有 runtime 字段或兼容别名，避免破坏已有分析脚本。

## 3. Consistency Validation

- [x] 3.1 增加 Multimodal-NF codebook `num_beam_classes` 与模型 beam head `num_classes` 的一致性校验。
- [x] 3.2 校验 objective、enabled heads、modalities 和 target schema 是否匹配，并在冲突时给出清晰错误。
- [x] 3.3 确保 `final_config.yaml`、`resolved_config.yaml`、`startup_summary.json` 和 `metrics.json` 中的 objective metadata 一致。

## 4. Tests

- [x] 4.1 使用 fixture 覆盖 Multimodal-NF near-field beam runtime metadata。
- [x] 4.2 使用 fixture 覆盖 Multimodal-NF LOS、link 和 selection multitask runtime metadata。
- [x] 4.3 添加 codebook 类别数不一致的拒绝测试。
- [x] 4.4 使用 `conda run -n kd_mm_beam pytest tests/test_multimodal_nf_dataset.py tests/test_prediction_objectives.py -q` 运行 focused 回归。
- [x] 4.5 视影响范围运行 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py -q` 中相关用例或新增 targeted tests。

## 5. Validation

- [x] 5.1 运行 `openspec validate clarify-multimodal-nf-runtime-semantics --strict`。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认 Purpose/TBD 和导入边界检查保持通过。
