## Why

当前 Scenario C 与 Scenario D 已经能给出 CxD 鲁棒性矩阵和 worst-case 指标，但论文核心 claim 仍缺少结构化证据来回答“JEPA 是否只是更好的 fusion，还是改变了模态主导关系”。需要把已有 accuracy table 升级为可审计的 phase transition、modality dominance 和 CNN/JEPA crossover 分析，使结果能直接支撑论文主图和 reviewer 追问。

## What Changes

- 新增 CxD phase transition analysis 能力，基于 `C0-C4` GPS async axis 与 `D0-D7` image observability axis 聚合 accuracy、DBA、robustness ratio、worst-case 和 robustness surface integral。
- 在 Scenario D benchmark 输出中增加机器可读的 phase diagram、dominance、crossover 和 failure-mode decomposition artifact，而不提交真实 metrics、figures、checkpoint 或 cache。
- 为支持模型输出 GPS/image/JEPA latent contribution score 的诊断契约，优先使用 gradient norm、attention/fusion weights、latent variance 等真实诊断；不可用时必须标记 skipped/unavailable，不得用启发式字段冒充解释性证据。
- 检测 CNN+GPS 与 Image-JEPA(+GPS/query-pool) 的 crossing region，并区分低退化 regime、高退化 regime、query_pool 相对 biased 的 crossover shift。
- 更新 smoke/fixture、runner manifest、图表生成和 focused tests，使 synthetic 测试能验证 schema、聚合、降级行为和 label 不移动 guard。
- 不新增训练入口、不恢复旧 KD/Hist/Top8/residual 路线、不把分析产物写入源码目录。

## Capabilities

### New Capabilities
- `cxd-phase-transition-analysis`: 定义 CxD phase diagram、modality dominance、crossing point、failure-mode decomposition 和论文图/表 artifact 的分析契约。

### Modified Capabilities
- `scenario-d-image-observability-benchmark`: 将 Scenario D CxD benchmark 的输出要求从基础 heatmap/图表扩展为可解释的 phase transition 与 dominance artifact。
- `jepa-gps-shortcut-benchmark`: 扩展 benchmark runner/manifest 对 CxD joint suite 的诊断字段、模型可比性和真实诊断降级要求。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/jepa_gps_shortcut_benchmark.py`、`src/kd_sensing/data/difficulty/*` 中的 C/D condition metadata 消费、`src/kd_sensing/models/*` 中可选 diagnostics 暴露，以及 `src/kd_sensing/cli/jepa_gps_shortcut_benchmark.py` 的只读 benchmark 输出。
- 需要更新 `configs/diagnostics/jepa_gps_shortcut_benchmark_scenario_d_smoke.yaml` 或新增同 family analysis manifest，用于声明模型组、C/D conditions、primary metric、diagnostic fallback policy 和输出 artifact plan。
- 需要补充 `tests/test_jepa_gps_shortcut_benchmark.py`、`tests/test_modality_difficulty.py` 或新增 focused tests，覆盖 CxD aggregation、dominance fallback、crossing detection、artifact manifest 和 no label shift。
- 需要同步 `docs/mainline_model_catalog.md`、`docs/experiment_protocols.md`、`docs/result_claims_registry.md`、`docs/experiment_matrix.md` 中 Scenario D / shortcut benchmark 的状态和 caveat；真实结果仍只记录本地产物路径与 claim status。
