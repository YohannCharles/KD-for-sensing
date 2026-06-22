## 1. 输入契约和数据汇总

- [x] 1.1 定义 evidence config schema，支持模型 pair、anchor baseline、P0-P5 metrics、benchmark manifest、forward cache 和输出目录字段。
- [x] 1.2 实现 evidence config 解析与 strict comparability 校验，写出 `evidence_manifest.json` 的基础 provenance。
- [x] 1.3 实现 P0-P5 wide/long 指标读取和规范化，保留 scene group、condition、metric、sample count 和来源路径。

## 2. Paired ablation 证据

- [x] 2.1 实现 GPS-query vs paired baseline 的 `paired_delta_by_condition.csv` 生成逻辑。
- [x] 2.2 实现 P0-P5 delta heatmap 和 Scene31/S32-S34 scene-group delta 图。
- [x] 2.3 将 strong non-JEPA/image+GPS baseline 标记为 anchor comparison，避免进入 paired claim gate。

## 3. 注意力热点图

- [x] 3.1 复用现有 attention collection，补齐 `[sample,time,query,patch]` 到 token grid 的聚合和 shape 校验。
- [x] 3.2 实现 patch-grid heatmap 输出，图中写入 model、sample id、scene、condition、target、Top-k 和 query aggregation。
- [x] 3.3 实现 image overlay 输出；缺少图像 tensor/path 或反归一化信息时降级为 patch-grid 并记录 warning。
- [x] 3.4 写出 `attention_summary.csv`，包含 entropy、effective patch count、query diversity 和 center-of-mass。

## 4. Case study 和报告

- [x] 4.1 实现 deterministic case selection，覆盖 `query_gain`、`query_regression`、`shared_near_miss` 和 `shared_failure`。
- [x] 4.2 导出 `tables/case_selection.csv`、`cases/*.json` 和 case panel 图。
- [x] 4.3 实现 claim gate summary，输出 `supported`、`exploratory`、`insufficient` 或 `blocked`。
- [x] 4.4 生成 `report.md`，明确区分 reportable、interpretive 和 caveat，并声明 attention 不是因果证明。

## 5. 配置、CLI 和文档

- [x] 5.1 增加最小 analysis/evidence config 示例，默认引用当前 CNN/hybrid sweep 的 ignored output 路径但不提交真实产物。
- [x] 5.2 将 evidence package 接入 `kd-sensing-jepa-visual-analysis` 的 opt-in 配置或新增薄 CLI 子入口。
- [x] 5.3 更新相关文档索引或实验说明，记录该证据包的用途、输入、输出和不能过度声称的边界。

## 6. 测试和验证

- [x] 6.1 添加 synthetic 单元测试，覆盖 paired delta、attention reshape、case selection、claim gate 和 unavailable fallback。
- [x] 6.2 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q`。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_cnn_hybrid_jepa_visual_prior_sweep.py -q`。
- [x] 6.4 运行 `openspec validate add-gps-query-attention-evidence-visuals --strict`。
