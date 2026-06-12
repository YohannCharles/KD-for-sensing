# jepa-visual-analysis-suite Specification

## Purpose
Define the supported offline JEPA visual-analysis workflow for comparing trained image/GPS beam-prediction models, exporting figures, tables, case payloads, cache metadata, and source-grounded reports without mutating training artifacts.
## Requirements
### Requirement: JEPA 可视化分析 CLI
系统 MUST 提供一个离线 JEPA 可视化分析入口，用于从分析配置读取多个模型的 config、checkpoint、评估 split 和图表开关，并将所有分析产物写入指定输出目录。该入口 MUST 不启动训练、不修改 checkpoint、不修改训练日志、不修改 split CSV。

#### Scenario: 运行分析 CLI
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --analysis-config <path> --output-dir <dir>`
- **THEN** 系统 MUST 读取分析配置并创建 `<dir>`
- **AND** 系统 MUST 对配置中声明的每个模型执行只读推理或读取已有 cache
- **AND** 系统 MUST 写出 `analysis_manifest.json`、`report.md`、`figures/`、`tables/` 和必要 `cache/` 产物

#### Scenario: 分析配置声明模型组
- **WHEN** 分析配置包含 `models.<name>.config` 和 `models.<name>.weights`
- **THEN** 系统 MUST 对每个模型使用对应 config 构建数据集和模型
- **AND** 系统 MUST 使用对应 weights strict load 或按配置记录非 strict load 结果
- **AND** 产物中的模型名 MUST 与配置中的 key 保持一致

#### Scenario: 分析入口只读训练产物
- **WHEN** 分析 CLI 读取已有训练 run、checkpoint 或评估目录
- **THEN** 系统 MUST 不修改该目录中的 `best.pth`、`last.pth`、`metrics.json`、`train_log.json`、`final_config.yaml` 或 split CSV
- **AND** 所有新增文件 MUST 写入分析输出目录

### Requirement: 评估协议复用与数据一致性
分析流程 MUST 复用项目现有评估数据构建、normalization artifact、checkpoint loading 和 forward 逻辑，确保可视化结果与正式评估使用同一 split、同一 scaler、同一 label space 和同一 `seq_len/num_pred`。

#### Scenario: 复用 2604 stratified split
- **WHEN** 模型配置启用 `stratified_80_10_10` 或等价 2604-style split
- **THEN** 分析流程 MUST 使用现有 protocol split 构建逻辑得到 test dataset
- **AND** `analysis_manifest.json` MUST 记录 scene、evaluation split、split protocol、split seed、`seq_len` 和 `num_pred`

#### Scenario: 指标与正式评估一致
- **WHEN** 分析流程从 logits 和 labels 计算 Top-k 与 DBA
- **THEN** 计算结果 MUST 与现有 evaluation metrics 在相同输入上的结果一致
- **AND** 若存在差异，系统 MUST 在 report 中标记 mismatch 并写出可调试的指标对比表

#### Scenario: metadata 安全清洗
- **WHEN** dataset metadata 包含 `None` 或 PyTorch DataLoader 默认 collate 不支持的值
- **THEN** 分析流程 MUST 只在分析 dataloader 内安全清洗 metadata
- **AND** 系统 MUST 不修改原始 dataset、CSV 或训练配置

### Requirement: 逐样本预测表
系统 MUST 为每个模型导出标准逐样本预测表，并 MUST 导出跨模型 join 后的对比表。逐样本表 MUST 至少包含样本标识、场景、target beam、Top-k 预测、target rank、Top-1 error、Top-3 min-distance、Top-5 min-distance、Top-10 hit、DBA contribution、entropy 和 margin 类诊断。

#### Scenario: 导出单模型逐样本表
- **WHEN** 分析流程完成某个模型的 test forward
- **THEN** 系统 MUST 写出 `tables/sample_predictions_<model>.csv`
- **AND** 表中每行 MUST 对应一个有效 test sample
- **AND** 表中 MUST 包含 target、top1、top3 list、top5 list、top10 list 和 sample-level DBA contribution

#### Scenario: 导出跨模型对比表
- **WHEN** 分析配置包含两个或更多模型
- **THEN** 系统 MUST 按稳定 sample id 或等价顺序 join 各模型逐样本预测
- **AND** 系统 MUST 写出 `tables/comparison_samples.csv`
- **AND** 表中 MUST 能识别 query gain、query regression、shared near miss 和 far error 样本组

#### Scenario: 缺少稳定 sample id
- **WHEN** dataset 未提供稳定 `sample_id`
- **THEN** 系统 MUST 使用 scene、split、global index 和 target beam 组成可追踪 fallback id
- **AND** `analysis_manifest.json` MUST 记录该 fallback 策略

### Requirement: 错误邻近性可视化
系统 MUST 导出用于解释 DBA 提升来源的错误邻近性图表和摘要，包括 Top-1 error histogram、Top-3 min-distance histogram、target rank distribution、DBA contribution distribution、beam confusion matrix、residual heatmap 和 per-target error/support scatter。

#### Scenario: 导出错误直方图
- **WHEN** `figures.error_anatomy` 启用
- **THEN** 系统 MUST 为每个模型导出 Top-1 error histogram 和 Top-3 min-distance histogram
- **AND** 图中 MUST 标注样本数、DBA delta、distance mode 和模型名

#### Scenario: 导出 rank 与 DBA contribution 图
- **WHEN** `figures.error_anatomy` 启用
- **THEN** 系统 MUST 导出 target rank in Top-10 的堆叠图或柱状图
- **AND** 系统 MUST 导出 sample-level DBA contribution 的 box plot、violin plot 或等价分布图

#### Scenario: 导出 beam-level 错误图
- **WHEN** `figures.error_anatomy` 启用
- **THEN** 系统 MUST 导出 target beam 到 predicted beam 的 confusion matrix
- **AND** 系统 MUST 导出 residual heatmap 或 residual histogram
- **AND** 系统 MUST 导出每个 target beam 的 support、Top-1、Top-3、DBA 或远错率摘要表

### Requirement: 表征空间可视化
系统 MUST 支持从配置指定的模型层或 diagnostics 中抽取 embedding，并导出表示空间可视化和邻域一致性指标。投影图 MUST 支持按 target beam、scene、error bucket 和模型名着色或分面。

#### Scenario: 抽取 embedding
- **WHEN** 分析配置声明 `embeddings.layers`
- **THEN** 系统 MUST 尝试从模型 diagnostics 或 forward hook 抽取对应层 embedding
- **AND** 系统 MUST 将 embedding、sample id 和标签信息写入 `cache/embeddings_<model>.npz`
- **AND** 若某层不存在，系统 MUST 记录 warning 并继续处理其他层

#### Scenario: 导出 UMAP/t-SNE/PCA 图
- **WHEN** `figures.embedding` 启用且至少一个模型存在可用 embedding
- **THEN** 系统 MUST 导出二维投影图
- **AND** 图中 MUST 记录实际使用的降维方法、random seed、样本数和 embedding 层名
- **AND** 若 UMAP 不可用，系统 MUST 自动降级到 t-SNE 或 PCA 并在 manifest 中记录降级原因

#### Scenario: 导出邻域一致性表
- **WHEN** embedding 可用
- **THEN** 系统 MUST 计算 kNN label purity 或 circular-neighbor consistency
- **AND** 系统 MUST 写出 `tables/embedding_neighbors.csv`
- **AND** report MUST 汇总每个模型的邻域 beam distance、同/邻 beam 比例和远错样本邻域统计

### Requirement: GPS-query attention 与图像显著性可视化
系统 MUST 对支持 attention diagnostics 的 GPS-query JEPA 模型导出 query-to-patch attention 可视化。对于没有 attention diagnostics 的 baseline，系统 MUST 安全降级为 probability/logit 曲线、embedding 对比或可选 Grad-CAM/显著性图，而不得让整个分析失败。

#### Scenario: 导出 GPS-query attention overlay
- **WHEN** 模型提供 `[batch,time,query,patch]` 或等价 GPS-query attention map
- **THEN** 系统 MUST 将 attention reshape 到 image patch grid
- **AND** 系统 MUST 为选中样本导出历史帧 attention overlay
- **AND** 图中 MUST 标注 history frame index、query index 或 averaged query/head、target beam 和 Top-k 预测

#### Scenario: 导出 attention 统计
- **WHEN** attention map 可用
- **THEN** 系统 MUST 计算 attention entropy、effective patch count、query diversity 和 attention center-of-mass
- **AND** 系统 MUST 写出 attention summary 表

#### Scenario: attention 不可用时安全降级
- **WHEN** 某模型不提供 attention diagnostics
- **THEN** 系统 MUST 跳过该模型的 attention overlay
- **AND** 系统 MUST 在 `analysis_manifest.json` 和 `report.md` 中记录 `attention_unavailable`
- **AND** 系统 MUST 继续生成其他图表

### Requirement: Case study 面板
系统 MUST 按 deterministic 规则选择并导出 qualitative case study 面板。Case group MUST 至少支持 `query_gain`、`query_regression`、`shared_near_miss` 和 `far_error`。

#### Scenario: 选择 query gain 样本
- **WHEN** 跨模型对比表中存在 baseline 远错而 GPS-query 模型 Top-3 near 或 hit 的样本
- **THEN** 系统 MUST 按固定 seed 和排序规则选择指定数量样本
- **AND** 系统 MUST 将选择依据写入 case selection 表

#### Scenario: 导出 case panel
- **WHEN** case study 样本被选中
- **THEN** 系统 MUST 导出包含输入 image strip、GPS 轨迹或 GPS 特征、各模型 Top-k 概率曲线、target beam、error 指标和可用 attention overlay 的面板
- **AND** 系统 MUST 为每个 case 写出机器可读 JSON payload

#### Scenario: 包含失败案例
- **WHEN** `far_error` group 启用
- **THEN** 系统 MUST 选择所有模型均远错或 GPS-query 模型远错的样本
- **AND** report MUST 将其标记为失败模式，而不得只展示成功案例

### Requirement: 轻量鲁棒性切片
系统 MUST 支持在不重新训练的情况下对模型进行 test-time robustness slicing，包括 drop image、drop GPS、GPS noise sweep 和 image masking sweep。鲁棒性图表 MUST 记录 clean baseline、扰动强度和 DBA/Top-k drop。

#### Scenario: 模态缺失切片
- **WHEN** `robustness.drop_modalities` 启用
- **THEN** 系统 MUST 至少评估 all、drop image 和 drop GPS 条件
- **AND** 系统 MUST 导出 `tables/robustness_summary.csv`
- **AND** 系统 MUST 导出 clean vs missing-modality DBA bar chart

#### Scenario: GPS noise sweep
- **WHEN** `robustness.gps_noise` 配置了一个或多个扰动强度
- **THEN** 系统 MUST 对 test GPS 输入施加 deterministic noise
- **AND** 系统 MUST 导出 DBA/Top-k 随噪声强度变化的曲线
- **AND** manifest MUST 记录 noise seed 和 noise 参数

#### Scenario: Image masking sweep
- **WHEN** `robustness.image_masking` 配置了一个或多个遮挡比例
- **THEN** 系统 MUST 对 test image 输入施加 deterministic mask
- **AND** 系统 MUST 导出 DBA/Top-k 随遮挡比例变化的曲线
- **AND** manifest MUST 记录 mask mode、mask seed 和遮挡比例

### Requirement: 报告与 manifest
系统 MUST 生成可直接阅读的 `report.md` 和机器可读 `analysis_manifest.json`。报告 MUST 总结模型对比、错误邻近性结论、表示空间结论、attention/case study 发现、鲁棒性发现和主要 caveat。

#### Scenario: 写出 analysis manifest
- **WHEN** 分析流程结束
- **THEN** 系统 MUST 写出 `analysis_manifest.json`
- **AND** manifest MUST 包含命令、分析配置路径或 digest、模型 config/weights、checkpoint load summary、split metadata、seed、输出文件清单和 warnings

#### Scenario: 写出 report
- **WHEN** 分析流程结束
- **THEN** 系统 MUST 写出 `report.md`
- **AND** report MUST 引用生成的主要图表路径和关键表格路径
- **AND** report MUST 包含“可报告结论”和“不能过度声称的 caveat”

#### Scenario: 图表格式
- **WHEN** 系统导出论文图表
- **THEN** 系统 MUST 至少支持 PNG 输出
- **AND** 系统 SHOULD 在可用时同时导出 SVG 或 PDF
- **AND** 图表 MUST 包含模型名、split、样本数和必要轴标签

### Requirement: 可测试性与降级行为
系统 MUST 为核心计算和产物 schema 提供自动化测试。缺少 optional dependency、缺少 attention、缺少 embedding layer 或样本数过少时，系统 MUST 记录 warning 并尽可能生成剩余产物。

#### Scenario: synthetic logits 指标测试
- **WHEN** 单元测试使用 synthetic logits、labels 和 metadata 调用分析指标 helper
- **THEN** Top-k、target rank、Top-3 min-distance 和 DBA contribution MUST 与预期值一致

#### Scenario: manifest schema 测试
- **WHEN** 分析流程在小型 mock 配置上完成
- **THEN** `analysis_manifest.json` MUST 包含必需字段
- **AND** 输出文件清单中的路径 MUST 存在或被标记为 skipped

#### Scenario: optional dependency 缺失
- **WHEN** UMAP、Grad-CAM 相关依赖或其他 optional visualization dependency 不可用
- **THEN** 系统 MUST 降级到可用方法或跳过对应图
- **AND** 系统 MUST 在 manifest 和 report 中记录 warning
- **AND** CLI MUST 以成功状态完成，除非所有必需输入均不可用
