## Context

当前实验已经得到一个值得写作的现象：`gps_query_pool_fasttrain` 在 S32/S33/S34 2604-style stratified 80/10/10 test 上提升 DBA，但 Top-1 exact accuracy 并不是最高。逐样本分析显示，大部分错误是近邻 beam 或 Top-3 内部排序问题，真正 Top-3 远离真值的样本很少。这意味着论文主张不应只停留在“指标更高”，而应证明 GPS-conditioned JEPA 表征改善了几何邻近结构、Top-k 候选排序和局部视觉聚合。

相关论文和常用可视化套路给出四类可借鉴证据：

- 表征学习论文常用 UMAP/t-SNE/PCA 投影展示 embedding 是否按语义或标签聚集；I-JEPA 强调非生成式 joint embedding 学习语义表征，可借鉴其“表征质量解释”思路。
- 多模态/BEV fusion 论文常用空间对齐、attention/feature map、跨模态贡献和 qualitative case study 说明融合机制；2604.05668 和 BEVFusion 都强调几何结构保留。
- 解释性视觉模型常用 Grad-CAM 或 attention overlay 展示图像区域是否与决策相关。
- missing-modality / robust multimodal fusion 论文常用 modality drop、noise sweep 和 performance degradation curve 证明部署鲁棒性。

本 change 将这些套路落到本项目的 beam prediction 任务：固定 fair protocol 和 checkpoint，导出可复现图表、逐样本表、聚合 summary 和报告，不改变训练主流程。

参考关键词与可借鉴来源：

- `I-JEPA`, `joint embedding predictive architecture`, `representation visualization`
- `BEVFusion`, `multi-modal sensor fusion`, `geometric structure`, `attention visualization`
- `DeepSense 6G`, `mmWave beam prediction`, `distance-based accuracy`, `Top-k beam prediction`
- `UMAP`, `t-SNE`, `Grad-CAM`, `attention heatmap`, `missing modality robustness`

## Goals / Non-Goals

**Goals:**

- 形成一套完整的“为什么 JEPA 更好”的可视化实验，覆盖指标、误差、表示、注意力、案例和鲁棒性。
- 支持同一套命令对比 `fair_base`、`fair_gps_biased`、`gps_query_pool`，并允许新增模型组。
- 所有图表必须能从 config/checkpoint/split 复现，并写出 manifest 记录输入、seed、样本选择和图表路径。
- 输出论文可直接使用的 PNG/SVG/PDF 图，以及用于补充材料的 CSV/JSON。
- 默认只读训练和评估产物；所有新输出写入用户指定分析目录。

**Non-Goals:**

- 不在本 change 中新增训练策略或改变模型结构。
- 不把 MMW 多天气适应性作为主线；最多预留后续扩展接口。
- 不强制引入 Gradio 交互页面；本 change 以离线、可批处理、可复现的论文图表为主。
- 不要求所有模型都必须提供 attention；没有 attention 的 baseline 使用 Grad-CAM、logit/error 或 embedding 对比安全降级。

## Decisions

### Decision 1: 以离线分析 CLI 为主，而不是直接扩展 Gradio viewer

新增一个离线分析入口，例如：

```bash
conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis \
  --analysis-config configs/diagnostics/jepa_visual_analysis_2604.yaml \
  --output-dir outputs/visual_analysis/jepa_query_pool_2604
```

分析配置声明模型组：

```yaml
models:
  fair_base:
    config: outputs/.../final_config.yaml
    weights: outputs/.../checkpoints/best.pth
  fair_gps_biased:
    config: outputs/.../final_config.yaml
    weights: outputs/.../checkpoints/best.pth
  gps_query_pool:
    config: outputs/fusion_image_gps_jepa_gps_query_pool_best_2604_s32_s34_fasttrain/final_config.yaml
    weights: outputs/fusion_image_gps_jepa_gps_query_pool_best_2604_s32_s34_fasttrain/checkpoints/best.pth
split:
  evaluation_split: test
  scenes: [32, 33, 34]
sampling:
  seed: 42
  max_embedding_samples: 3000
  case_groups:
    - query_gain
    - query_regression
    - shared_near_miss
    - far_error
figures:
  embedding: true
  error_anatomy: true
  attention: true
  case_studies: true
  robustness: true
```

理由：论文图需要批量复现、可缓存、可在无浏览器环境运行；Gradio 更适合后续人工浏览。备选方案是把所有分析塞进 viewer，但会混合交互逻辑和论文产物逻辑，测试和复现更难。

### Decision 2: 抽取统一 per-sample analysis table 作为所有图表的根数据

每个模型先导出标准逐样本表：

- `sample_id`、scene、split、原始 CSV 来源、全局 index
- target beam、Top-1/Top-3/Top-5/Top-10、target rank
- Top-1 linear/circular error、Top-3 min-distance、DBA contribution
- logits/probability summary：entropy、margin、GT probability、top1-top2 margin
- optional metadata：GPS path、image path、sequence/frame 信息

跨模型 join 后形成 `comparison_samples.csv`，用于 gain/regression/case selection。

理由：先有表，再画图，可以保证每张图都能追溯样本来源，也能避免不同图各自重跑推理导致顺序或过滤不一致。备选方案是每个图直接访问 dataloader 和模型，但容易引入重复推理和不可复现差异。

### Decision 3: 可视化分成六个“证据块”

#### A. 主指标与公平口径面板

图表：

- S32/S33/S34 DBA bar + macro DBA
- combined Top-1/Top-3/Top-5/DBA table
- 与 2604 BEV-Fusion 报告值的虚线参考线

回答的问题：我们是否在同一复现实验口径下真的更强。

#### B. 错误邻近性与 DBA 机制面板

图表：

- Top-1 error histogram：展示 exact miss 大多是 ±1/±2。
- Top-3 min-distance histogram：展示 DBA 高的直接原因。
- target rank in Top-10 stacked bar：展示真值常在 rank 2/3。
- DBA contribution violin/box plot：展示每个模型的样本级 DBA 分布。
- beam confusion matrix 与 residual heatmap：横轴 target beam，纵轴 predicted beam 或 residual。
- per-target beam support vs error scatter：找少样本或远错 beam。

回答的问题：提升来自“更近的错”和“Top-k 排序更合理”，而不是只靠少数 lucky hit。

#### C. 表征空间面板

图表：

- UMAP/t-SNE/PCA 2D 投影：分别抽取 image/JEP A context/fusion/temporal embedding，按 target beam、scene、error bucket 着色。
- 邻近 beam 连续性图：embedding kNN 的平均 beam distance 分布。
- label purity / circular-neighbor consistency 表：kNN 中相邻 beam 比例、同 scene 混合程度、远错样本局部密度。
- trajectory arrows：同一序列历史 5 帧 embedding 的移动方向与 target beam residual 的关系。

回答的问题：JEPA 是否让表示空间更符合 beam 几何邻近关系。

实现方式：优先通过 model diagnostics 或 forward hook 抽取配置指定层；若 UMAP 可用则使用 UMAP，否则降级到 t-SNE/PCA，并在 manifest 中记录降级原因。

#### D. GPS-query attention / 图像显著性面板

图表：

- GPS query 到 image patch tokens 的 attention overlay，按历史 5 帧展示。
- query diversity：每个 query 的 attention entropy、effective patch count、center-of-mass。
- temporal attention drift：attention center 随历史帧移动的轨迹，与 GPS 运动方向对齐。
- baseline Grad-CAM 或 saliency overlay：给没有 GPS-query attention 的模型提供可比可解释图。

回答的问题：GPS-query pooling 是否真的在用 GPS 条件选择图像局部区域，而不是退化成均值池化。

注意：attention 只作为诊断，不反向参与训练。若 checkpoint 没有保存 attention 开关，则分析命令应允许通过 config override 启用诊断，或安全跳过并记录 warning。

#### E. 论文 case study 面板

每个 case panel 包含：

- 5 帧输入 image strip 与 GPS 轨迹小图
- target beam 与各模型 Top-5 bar/probability curve
- JEPA attention overlay
- baseline vs JEPA 的 Top-1 error、Top-3 min-distance、DBA contribution
- 简短自动生成 caption：`query_gain`、`shared_near_miss`、`regression`、`far_error`

样本选择必须 deterministic：

- `query_gain`: baseline Top-3 far，gps_query_pool Top-3 near 或 hit。
- `query_regression`: baseline near，gps_query_pool far。
- `shared_near_miss`: 两者 Top-1 均错，但 gps_query_pool rank 更靠前或 DBA 更高。
- `far_error`: 所有模型均远错，用来诚实展示失败模式。

回答的问题：定性图是否与定量 claim 一致，且不 cherry-pick。

#### F. 轻量鲁棒性切片

图表：

- drop image / drop GPS / all modalities 的 DBA bar。
- GPS noise sweep：不同标准差或米级扰动下 DBA 曲线。
- image masking sweep：遮挡比例或中心/随机 patch mask 下 DBA 曲线。
- robustness delta table：相对 clean 的 DBA drop。

回答的问题：JEPA 表征是否更稳，是否有部署价值。

非目标：不在这里做 MMW 多天气大故事；如果后续需要，可复用同一 sweep 框架扩展到天气/domain labels。

### Decision 4: 产物目录与 manifest 标准化

输出目录建议：

```text
outputs/visual_analysis/<analysis_name>/
  analysis_manifest.json
  report.md
  figures/
    01_metric_panel.svg
    02_error_anatomy.svg
    03_embedding_umap_target_beam.svg
    04_embedding_error_bucket.svg
    05_attention_cases/
    06_robustness_curves.svg
  tables/
    model_metrics.csv
    sample_predictions_<model>.csv
    comparison_samples.csv
    embedding_neighbors.csv
    robustness_summary.csv
  cache/
    embeddings_<model>.npz
    logits_<model>.npz
```

`analysis_manifest.json` 必须记录：

- analysis config digest、代码版本信息、命令行参数
- 每个模型的 config、weights、checkpoint load summary
- split protocol、scene、seq_len、num_pred、evaluation_split
- 图表开关、采样 seed、实际样本数
- 每个输出文件的相对路径和生成时间

### Decision 5: 使用现有评估构建逻辑，避免旁路数据协议

分析模块应复用现有 `build_protocol_split_datasets`、`build_dataloader`、`build_model`、`load_model_state`、`run_evaluation_pass` 逻辑。必要时对 metadata 做只读清洗，但不得绕过 2604 split、GPS scaler、image cache policy 或 checkpoint strict load。

理由：可视化必须与正式评估完全同口径。备选方案是直接读 CSV 和图片路径再手写 forward，但容易产生 split/normalization 不一致。

### Decision 6: 图表默认保存矢量格式，同时保留 PNG preview

论文图默认保存 SVG 或 PDF，另存 PNG 便于快速查看。所有图必须包含标题、轴标签、单位、模型名、split 标记和样本数；颜色映射应固定，避免不同图间含义漂移。

## Risks / Trade-offs

- [Risk] UMAP/t-SNE 可能被误解为严格定量证明。→ Mitigation：同时导出 kNN label purity、circular-neighbor consistency 和 embedding neighbor beam distance 等表格指标，并在报告中说明投影仅作可视化。
- [Risk] attention heatmap 不等于因果解释。→ Mitigation：将其定位为诊断证据；同时提供 masking/ablation 或 Grad-CAM 辅助验证。
- [Risk] case study 被认为 cherry-pick。→ Mitigation：样本选择规则 deterministic，输出完整候选表和每类固定数量，包含 gain、regression、far error。
- [Risk] baseline 不提供 attention，导致可视化不公平。→ Mitigation：attention 只用于解释 GPS-query 模块；baseline 用概率曲线、Grad-CAM 或 embedding 对比，不强求同种 heatmap。
- [Risk] 逐模型抽 embedding 耗时或显存高。→ Mitigation：支持 `max_embedding_samples`、batch-size override、cache reuse、CPU 降维、按图开关。
- [Risk] 额外依赖污染训练环境。→ Mitigation：UMAP 设为 optional；没有 `umap-learn` 时降级到 sklearn t-SNE/PCA；核心训练依赖不新增强制 UI 包。
- [Risk] 图表与正式评估数字不一致。→ Mitigation：manifest 记录评估复算结果，并在测试中用 synthetic logits 校验 Top-k/DBA 计算与现有 metrics 一致。

## Migration Plan

1. 新增离线分析 CLI 与诊断模块，默认不被训练/评估流程调用。
2. 新增 2604 Image+GPS+JEPA 的示例分析配置，路径使用用户本地 checkpoint 时允许占位。
3. 在 README 或实验文档中添加运行命令和产物说明。
4. 若实现过程中需要新增 optional dependency，只加入 visualization/diagnostics extra 或文档提示。
5. 回滚时删除新增 CLI、诊断模块和示例配置即可；训练、评估和 checkpoint 格式不受影响。

## Open Questions

- 当前 `fair_base` 和 `fair_gps_biased` 的最终 checkpoint 路径是否已有固定命名；若没有，示例配置应使用占位并要求用户传入。
- 是否把 attention diagnostics 默认写入模型 forward diagnostics，还是只在分析 CLI 中通过 hook 临时开启。
- 论文主图优先使用 SVG/PDF 还是 PNG；建议默认三者都可选，SVG/PDF 为主。
