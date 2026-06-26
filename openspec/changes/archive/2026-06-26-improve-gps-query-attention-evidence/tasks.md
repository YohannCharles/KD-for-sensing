## 1. 诊断契约与配置

- [x] 1.1 在 JEPA visual analysis/evidence config 解析中增加 attention faithfulness opt-in 配置，包括 patch ratio/count、selection groups、occlusion strategy、random seed、max cases 和 metric target。
- [x] 1.2 定义 attention provenance 输出字段，覆盖 `map_semantics`、`causal_claim`、attention source、tensor shape、token grid、aggregation method、normalization scope、overlay image source 和 cross-sample comparability。
- [x] 1.3 保留现有 attention summary 与 overlay 字段兼容，只追加新字段，不迁移旧本地产物。

## 2. Pooler 诊断 metadata

- [x] 2.1 扩展 `GPSQueryPool` diagnostics，记录 `attention_head_aggregation=averaged`、attention 输出 shape、query count、token count 和 condition feature source。
- [x] 2.2 扩展 hybrid/predictive GPS-query 类 pooler diagnostics，分别记录 content/GPS branch attention summary、unavailable reason 和 `last_attention_map` 暴露策略。
- [x] 2.3 增加 opt-in per-head attention diagnostics 支持，默认保持 averaged attention shape 和训练 forward 主输出不变。
- [x] 2.4 更新 `tests/test_gps_conditioned_jepa.py` 或相邻 focused tests，覆盖默认 averaged 兼容、per-head metadata 和 branch summary。

## 3. Attention faithfulness 核心计算

- [x] 3.1 实现 deterministic patch selection helper，支持 `top_attention`、`low_attention` 和 `random`，并验证三组 patch budget 一致。
- [x] 3.2 实现 image/tensor 级遮挡 helper，支持 `zero` 和可用时的 `dataset_mean`，并在缺少可遮挡输入时返回 structured skipped reason。
- [x] 3.3 复用现有只读 forward/metric helper 计算 baseline 与 occluded 的 target logit、target margin、Top-k、DBA contribution 或配置指定等价指标。
- [x] 3.4 写出 `tables/attention_faithfulness.csv`，包含 model、sample id、selection group、patch budget、occlusion strategy、seed、baseline metric、occluded metric、delta 和 faithfulness status。

## 4. Evidence package 与 report 集成

- [x] 4.1 将 attention provenance 和 faithfulness summary 写入 `analysis_manifest.json` 或 `evidence_manifest.json`，包含 skipped reason 和截断样本数。
- [x] 4.2 在 claim gate 中接入 faithfulness 结果，保证 paired delta 和 strict comparability 优先于 attention 解释项。
- [x] 4.3 更新 `report.md` 生成逻辑，单独汇总 token-read map、faithfulness 结果、query diversity/effective patch count、失败样本和 caveat。
- [x] 4.4 在可视化依赖可用时导出 `figures/attention_faithfulness/` 简洁图表；依赖不可用时保留 CSV 并记录 skipped reason。

## 5. 测试与验证

- [x] 5.1 更新 `tests/test_jepa_visual_analysis.py`，使用 synthetic attention/logits/image tensor 覆盖 provenance、faithfulness CSV、report caveat 和 unavailable fallback。
- [x] 5.2 更新 GPS-query evidence 相关 tests，覆盖 claim gate 在 faithfulness 通过、不通过、paired evidence 不足三种情况下的判定。
- [x] 5.3 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py tests/test_gps_conditioned_jepa.py -q`。
- [x] 5.4 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`，确认未恢复 viewer/旧 visualization 入口。
- [x] 5.5 运行 `openspec validate improve-gps-query-attention-evidence --strict`。
