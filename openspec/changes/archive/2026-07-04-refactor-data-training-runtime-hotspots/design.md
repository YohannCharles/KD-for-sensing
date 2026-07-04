## Context

训练链路热点影响面最大：DeepSense6G dataset 影响大量训练 IO 测试，MMW dataset 继承 DeepSense6GDataset 并叠加 family adapter，training/evaluation runtime 同时服务 supervised、JEPA、difficulty、prediction objective 和 MMW GPS v2 workflow。该 change 必须按 wave 小步移动，避免隐式改变数据 split、beam label、loss 或 checkpoint 行为。

## Goals / Non-Goals

**Goals:**
- 将 dataset contract、resource reader、target provider、scaler/normalizer、family adapter 和 physical label 拆到稳定 owner。
- 将训练 runtime 的 context/resource/loop/finalize 阶段边界清晰化。
- 将 batch/evaluation pass 的 target、metadata、objective output 和 metric aggregation 拆分。
- 将 MMW GPS v2 workflow 的 protocol、support selection 和 artifact writer 解耦。

**Non-Goals:**
- 不改变训练默认值、优化器、AMP、early stopping、checkpoint policy。
- 不改变 dataset split、beam label mapping、soft target、no-future-leak 规则。
- 不读取真实数据做验证。

## Decisions

1. **按风险从低到高实施。**
   先抽纯 helper 和 writer，再动 dataset `__getitem__`、training loop 或 evaluation aggregation。

2. **dataset contract helper 优先。**
   新的 GPS feature mode、beam target source、column guard、cache path rule 必须进入 contract/cache/target helper，而不是扩大 `DeepSense6GDataset` 主体。

3. **training runtime 保持阶段函数。**
   `_train_inner` 已较薄，后续重点是拆 `_prepare_training_run_context`、resource build 和 finalization，不把 loop 细节塞回入口函数。

4. **evaluation schema 通过 fixture 固定。**
   每次拆分都先保留 `EvaluationPassResult`、prediction metadata、objective runtime metadata 和 metric keys。

## Risks / Trade-offs

- Dataset 影响面大 -> 使用 synthetic tests 覆盖 cache、label、target、scaler 和 lazy loading，不读取真实 `dataset/`。
- 训练数值语义漂移 -> 不合并行为修改；每个 wave 运行 focused tests，并保留 full regression 作为最终验收。
- MMW GPS v2 artifact 字段漂移 -> 对 summary/prediction/support/theta/branch rows 做 schema 断言。

## Migration Plan

1. 更新 inventory 中 dataset/training/runtime 当前规模与 split-next 方向。
2. 先拆 DeepSense6G/MMW 纯 helper，再拆 evaluation/batch helper。
3. 抽 MMW GPS v2 artifact writer 与 support selection helper。
4. 最后整理 training context/resource/finalize。
5. 运行 `openspec validate refactor-data-training-runtime-hotspots --strict`、`conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_deepsense6g_contract_helpers.py tests/test_mmw_town10_preparation.py tests/test_evaluation_pass.py tests/test_prediction_objectives.py tests/test_architecture_boundaries.py -q`。

## Open Questions

- `MMWDataset` 是否继续继承 `DeepSense6GDataset`，还是逐步提取 shared sequence dataset base？
- `EvaluationPassResult` 是否需要 dataclass 拆分 objective-specific payload？
