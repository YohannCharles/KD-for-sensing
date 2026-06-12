## 1. 配置与入口

- [x] 1.1 新增 JEPA visual analysis 配置 schema/helper，支持 models、split、sampling、figures、robustness、outputs 字段。
- [x] 1.2 新增离线 CLI 入口 `kd-sensing-jepa-visual-analysis`，支持 `--analysis-config`、`--output-dir`、`--override`、`--force` 和 `--dry-run`。
- [x] 1.3 新增示例配置 `configs/diagnostics/jepa_visual_analysis_2604.yaml`，包含 `fair_base`、`fair_gps_biased`、`gps_query_pool` 占位或当前可用路径。
- [x] 1.4 确保 CLI 默认只读训练/评估产物，并将所有输出写入指定 analysis output directory。

## 2. 数据构建与逐样本预测表

- [x] 2.1 复用现有 evaluation dataset/model/checkpoint 构建逻辑，封装单模型只读 forward runner。
- [x] 2.2 实现 metadata 安全清洗 wrapper，处理 `None`、嵌套 dict/list 和 DataLoader collate 兼容问题。
- [x] 2.3 实现 per-sample metric helper，计算 Top-k、target rank、Top-1 error、Top-3/5/10 min-distance、DBA contribution、entropy、margin 和 GT probability。
- [x] 2.4 为每个模型写出 `tables/sample_predictions_<model>.csv`、logits/probability cache 和 metrics summary。
- [x] 2.5 实现跨模型 sample join，写出 `tables/comparison_samples.csv`，并标注 `query_gain`、`query_regression`、`shared_near_miss`、`far_error`。

## 3. 错误邻近性图表

- [x] 3.1 实现 Top-1 error histogram 和 Top-3 min-distance histogram 导出。
- [x] 3.2 实现 target rank in Top-10 分布图和 DBA contribution 分布图。
- [x] 3.3 实现 beam confusion matrix、residual heatmap 和 per-target support/error scatter。
- [x] 3.4 为错误图表写出对应 CSV summary，并在图中标注模型名、scene/split、样本数、distance mode 和 DBA delta。

## 4. 表征空间分析

- [x] 4.1 实现 embedding 抽取配置与 forward hook/diagnostics adapter，支持缺失层 warning。
- [x] 4.2 写出 `cache/embeddings_<model>.npz`，包含 embedding、sample id、target、scene、error bucket 和 layer name。
- [x] 4.3 实现 UMAP/t-SNE/PCA 降维流程，UMAP 缺失时自动降级并记录 manifest warning。
- [x] 4.4 导出按 target beam、scene、error bucket、model 分面或着色的 embedding 图。
- [x] 4.5 实现 kNN label purity、circular-neighbor consistency 和 embedding neighbor beam distance，写出 `tables/embedding_neighbors.csv`。

## 5. Attention、显著性与 Case Study

- [x] 5.1 接入 GPS-query attention diagnostics，支持 `[B,T,K,N]` 或等价 attention map reshape 到 patch grid。
- [x] 5.2 导出 GPS-query attention overlay、query entropy、effective patch count、query diversity 和 attention center-of-mass 表。
- [x] 5.3 对无 attention 的 baseline 实现安全降级，至少保留 probability/logit 曲线和 embedding/error 对比；可选接入 Grad-CAM 或显著性图。
- [x] 5.4 实现 deterministic case selection，支持 `query_gain`、`query_regression`、`shared_near_miss`、`far_error`。
- [x] 5.5 导出 case study panel，包含 5 帧 image strip、GPS 轨迹或特征、各模型 Top-k 曲线、target beam、error 表和可用 attention overlay。
- [x] 5.6 为每个 case 写出机器可读 JSON payload 和 case selection CSV。

## 6. 鲁棒性切片

- [x] 6.1 实现 test-time drop image、drop GPS 和 all modality 对照评估，复用模型 force modality mask 或安全输入扰动机制。
- [x] 6.2 实现 deterministic GPS noise sweep，记录 noise seed、强度和输入空间。
- [x] 6.3 实现 deterministic image masking sweep，记录 mask seed、mask mode 和遮挡比例。
- [x] 6.4 写出 `tables/robustness_summary.csv`，导出 clean vs disturbed DBA/Top-k bar chart 和 sweep curve。

## 7. 报告、Manifest 与文档

- [x] 7.1 实现 `analysis_manifest.json` writer，记录命令、配置 digest、模型路径、checkpoint load summary、split metadata、seed、warnings 和输出文件清单。
- [x] 7.2 实现 `report.md` writer，总结主指标、错误邻近性、embedding、attention/case、鲁棒性和 caveat。
- [x] 7.3 在图表导出中支持 PNG，并在可用时支持 SVG/PDF；所有图包含轴标签、样本数、split 和模型名。
- [x] 7.4 更新实验文档或 README，说明如何运行分析命令、如何解释图表、如何引用 caveat。

## 8. 测试与验证

- [x] 8.1 添加 synthetic logits 单元测试，验证 Top-k、rank、Top-3 min-distance 和 DBA contribution 与现有 metrics 一致。
- [x] 8.2 添加 analysis config 解析和 override 测试。
- [x] 8.3 添加 manifest schema 和输出路径清单测试。
- [x] 8.4 添加 optional dependency 缺失、attention 缺失、embedding layer 缺失的降级测试。
- [x] 8.5 运行 `conda run -n kd_mm_beam pytest tests/test_gps_conditioned_jepa.py -q` 确认 JEPA 相关既有行为不回退。
- [x] 8.6 运行新增分析测试，例如 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q`。
- [x] 8.7 运行 `openspec validate add-jepa-visual-analysis-suite --strict` 和 `openspec status --change add-jepa-visual-analysis-suite`。
