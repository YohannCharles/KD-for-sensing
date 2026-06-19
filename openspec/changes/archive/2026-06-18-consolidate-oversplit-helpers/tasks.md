## 1. 实现前审计与合并清单

- [x] 1.1 阅读 `docs/agent_navigation.md`、`docs/maintainer_context_index.yaml`、本 change 的 `proposal.md`、`design.md` 和 specs，确认本次只做内部 helper 合并与冗余检查精简，不改训练/评估语义
- [x] 1.2 用当前源码 imports 和 focused tests 建立合并清单，标记每个候选 helper 的 owner、调用方、是否 public import、是否单调用点和是否可删除
- [x] 1.3 记录必须保留的边界检查清单：manifest/config/CLI、路径存在性、split/label-space/metric comparability、no-future-leak、difficulty replay metadata 和本地产物边界
- [x] 1.4 确认 active change `add-predictive-gps-query-advantage` 的 JEPA predictive 语义不被本 change 修改

## 2. JEPA Benchmark Helper 合并

- [x] 2.1 将 `jepa_benchmark_common_types.py`、`jepa_benchmark_io.py`、`jepa_benchmark_metadata.py`、`jepa_benchmark_scalars.py` 中仍被使用的 helper 合并回 `src/kd_sensing/diagnostics/jepa_benchmark_common.py`
- [x] 2.2 更新所有 `jepa_benchmark_common_*` import，删除不再需要的 common helper 文件，并保持 `BenchmarkManifestError`、warning record、JSON/CSV/scalar helper 行为兼容
- [x] 2.3 将 Scenario D/CxD 的 normalization、metrics、phase、dominance、failure-mode 和 pairing helper 合并到 `jepa_benchmark_scenario_d.py` 或一个明确的 Scenario D owner 模块
- [x] 2.4 更新 Scenario D/CxD imports 和 public re-export，删除不再需要的 `jepa_benchmark_cxd_*.py`、`jepa_benchmark_scenario_d_metrics.py`、`jepa_benchmark_scenario_d_normalization.py`
- [x] 2.5 将 runner summary、metric source ingestion 和 runner manifest helper 合并回 `jepa_benchmark_runner.py`
- [x] 2.6 更新 runner imports，删除不再需要的 `jepa_benchmark_runner_summary.py`、`jepa_benchmark_runner_sources.py`、`jepa_benchmark_runner_manifest.py`
- [x] 2.7 保持 `jepa_gps_shortcut_benchmark.py` 为 thin public facade，不把 runner、Scenario C/D、plotting 或 artifact registry 主体实现合回 facade
- [x] 2.8 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`

## 3. Data Factory、Sequence 和 BeamBench 合并

- [x] 3.1 将 `data_factory_validation.py` 中只服务 dataloader build flow 的 helper 合并回 `src/kd_sensing/engine/data_factory.py`
- [x] 3.2 审核 `data_factory_scalers.py`、`data_factory_loaders.py`、`data_factory_groups.py`、`data_factory_protocols.py`，仅合并单调用点或低复用 helper，保留仍有清晰 owner 价值的 split/scaler/protocol 模块
- [x] 3.3 更新 data factory imports 和 `__all__`，删除已完全回收的 helper 文件
- [x] 3.4 审核 `sequence_columns.py`、`sequence_metadata.py`、`sequence_splits.py`、`sequence_windows.py`，仅合并低复用 helper，保持窗口生成、split 和 metadata 语义可读
- [x] 3.5 审核 `image_ae_gps_*` 文件，优先回收薄 wrapper、reports/evaluation 小 helper，保留训练、dataset、model 中仍明显内聚的 owner 模块
- [x] 3.6 按触碰范围运行 focused tests，例如 `conda run -n kd_mm_beam pytest tests/test_training_io_workflow.py tests/test_epoch_subsampling.py -q` 和 `conda run -n kd_mm_beam pytest tests/test_beambench_image_ae_gps_direct.py -q`

## 4. 冗余防御检查精简

- [x] 4.1 删除内部私有 helper 中由调用方已保证的重复 `assert`、重复 `isinstance`、重复 `None` 检查和同义异常包装
- [x] 4.2 保留 manifest/config/CLI、路径解析、comparability、no-future-leak、difficulty replay metadata、输出产物边界相关检查
- [x] 4.3 确保删除检查后失败会自然暴露为底层 Python/PyTorch/IO 错误，且不会吞掉 manifest warning 或 schema 字段
- [x] 4.4 对涉及指标或 row schema 的精简补跑 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`

## 5. 治理索引、文档和架构测试同步

- [x] 5.1 更新 `docs/maintainer_context_index.yaml` 的 hotspot 文件列表、line budget、status、planned_action、consolidation_targets 和 validation commands
- [x] 5.2 更新 `docs/project_surface_inventory.md` 中对应 right-size / merge-candidate / accepted-size 说明
- [x] 5.3 更新 `tests/test_architecture_boundaries.py` 中 helper 文件存在性、forbidden snippets 和 max_lines 期望，使其匹配合并后的 owner 模块
- [x] 5.4 如 README 或 `docs/agent_navigation.md` 中有旧 helper 路由说明，同步改为合并后的 owner 路由

## 6. 最终验证

- [x] 6.1 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_gps_shortcut_benchmark.py -q`
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- [x] 6.3 按实际触碰范围运行 data factory、sequence 和 BeamBench focused tests
- [x] 6.4 运行 `conda run -n kd_mm_beam kd-sensing-jepa-gps-shortcut-benchmark --help`
- [x] 6.5 运行 `openspec validate consolidate-oversplit-helpers --strict`
- [x] 6.6 汇总最终删除的文件、保留的必要检查、已运行验证、未运行验证和剩余风险
