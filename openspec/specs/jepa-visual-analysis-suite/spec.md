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
系统 MUST 对支持 attention diagnostics 的 GPS-query JEPA 模型导出 query-to-patch attention 可视化，并 MUST 在 image modality 可用时额外导出映射到原始 RGB 图片大小的 attention heatmap overlay。对于没有 attention diagnostics 或没有可用图像输入的 baseline，系统 MUST 安全降级为 probability/logit 曲线、embedding 对比、patch-grid attention 或可选 Grad-CAM/显著性图，而不得让整个分析失败。

#### Scenario: 导出 GPS-query attention patch grid
- **WHEN** 模型提供 `[batch,time,query,patch]` 或等价 GPS-query attention map
- **THEN** 系统 MUST 将 attention reshape 到 image patch grid
- **AND** 系统 MUST 为选中样本导出历史帧/query attention patch-grid 或 query-time 分面图
- **AND** 图中 MUST 标注 history frame index、query index 或 averaged query/head、target beam 和 Top-k 预测

#### Scenario: 导出 image-space attention overlay
- **WHEN** 模型提供 GPS-query attention map 且对应样本 metadata 能解析到原始 RGB 图片
- **THEN** 系统 MUST 读取原始 RGB 图片
- **AND** 系统 MUST 将 14x14 或等价 patch-grid attention 上采样到原始图片 height/width
- **AND** 系统 MUST 将原始图片处理为低对比底图后叠加彩色 attention heatmap
- **AND** 系统 MUST 为选中样本写出 `figures/attention_image_overlays/` 下的 PNG 或配置声明格式图像
- **AND** overlay 图 MUST 标注模型名、sample id 或 index、history frame index、query index、target beam、Top-k 预测、attention 归一化方式、overlay 样式和底图来源

#### Scenario: 原始图片缺失时使用模型输入图兜底
- **WHEN** attention diagnostics 可用但样本原始 RGB 图片路径缺失或不可读
- **THEN** 系统 MAY 使用模型输入 RGB tensor 作为 overlay 底图
- **AND** overlay 图或 manifest MUST 标记 `overlay_image_source=model_input_tensor`
- **AND** 系统 MUST 不因原始图片缺失而中断其它 attention 图表生成

#### Scenario: overlay 归一化和 caveat
- **WHEN** 系统导出 image-space attention overlay
- **THEN** 同一样本内的 time/query 子图 MUST 使用共享颜色尺度或在图中明确标注使用的归一化方式
- **AND** `analysis_manifest.json` 或 `report.md` MUST 记录 overlay 使用的是原始图片大小或模型输入图兜底
- **AND** 报告 MUST 不把 raw attention overlay 单独描述为因果解释或严格归因证据

#### Scenario: 默认论文式 overlay 样式
- **WHEN** 用户未显式配置 overlay 样式
- **THEN** 系统 MUST 使用默认 `paper_overlay` 样式
- **AND** 默认样式 MUST 保留原图结构但降低底图视觉竞争
- **AND** 默认样式 MUST 使用蓝-绿-黄-红连续热力图突出高 attention 区域

#### Scenario: 导出 attention 统计
- **WHEN** attention map 可用
- **THEN** 系统 MUST 计算 attention entropy、effective patch count、query diversity 和 attention center-of-mass
- **AND** 系统 MUST 写出 attention summary 表

#### Scenario: attention 或 image 不可用时安全降级
- **WHEN** 某模型不提供 attention diagnostics
- **THEN** 系统 MUST 跳过该模型的 attention overlay
- **AND** 系统 MUST 在 `analysis_manifest.json` 和 `report.md` 中记录 `attention_unavailable`
- **AND** 系统 MUST 继续生成其他图表
- **WHEN** attention diagnostics 可用但样本没有可用原始 RGB 图片且没有可用模型输入图兜底
- **THEN** 系统 MUST 跳过 image-space overlay
- **AND** 系统 MUST 继续生成 patch-grid attention 图和 attention summary 表
- **AND** 系统 MUST 记录 `attention_image_overlay_unavailable`

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

### Requirement: Benchmark perturbation manifest 分析输入
JEPA visual analysis MUST 能读取 JEPA GPS shortcut benchmark manifest 或 benchmark runner 输出的机器可读 manifest，并将其中声明的模型、扰动 suite、severity、seed、split metadata 和指标产物纳入离线分析。分析入口 MUST 保持只读训练产物。

#### Scenario: 读取 benchmark manifest
- **WHEN** 用户运行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --analysis-config <path>` 且分析配置引用 benchmark manifest
- **THEN** 分析流程 MUST 读取 benchmark 的模型列表、扰动条件、severity sweep、metrics 表和 warnings
- **AND** 输出的 `analysis_manifest.json` MUST 记录 benchmark manifest 路径或 digest
- **AND** 分析流程 MUST 不修改 benchmark 输入表、训练 checkpoint、训练日志或 split CSV

#### Scenario: benchmark manifest 缺少可选图表输入
- **WHEN** benchmark manifest 未提供 attention、embedding 或 case payload 所需字段
- **THEN** 分析流程 MUST 跳过对应图表
- **AND** `analysis_manifest.json` 和 `report.md` MUST 记录 skipped reason
- **AND** 已具备输入的鲁棒性表和曲线 MUST 继续生成

### Requirement: Benchmark robustness matrix 图表和表格
JEPA visual analysis MUST 支持从 benchmark 指标表生成跨模型、跨扰动 suite、跨 severity 的 robustness matrix、collapse curve 和 clean-delta summary。输出 MUST 保持与现有 `figures/`、`tables/` 和 `report.md` 结构一致。

#### Scenario: 导出跨模型 robustness matrix
- **WHEN** benchmark 指标表包含两个或更多模型和一个或更多扰动 suite
- **THEN** 分析流程 MUST 写出跨模型 robustness matrix 表
- **AND** 表中 MUST 包含 clean metric、perturbed metric、delta、relative drop、sample_count、suite、condition 和 severity

#### Scenario: 导出 GPS collapse 曲线
- **WHEN** benchmark 指标表包含 GPS noise、GPS missing、GPS drift 或 GPS distractor suite
- **THEN** 分析流程 MUST 导出对应 collapse curve
- **AND** 图表 MUST 标注模型名、severity 单位、metric、split、样本数和 seed 或 digest

#### Scenario: 导出 image degradation 曲线
- **WHEN** benchmark 指标表包含 fog/rain、night、occlusion 或 motion blur suite
- **THEN** 分析流程 MUST 导出 image degradation robustness curve
- **AND** 图表 MUST 将 physical degradation type 与普通 augmentation 说明区分记录在 metadata 中

### Requirement: Shortcut reliance 报告段落
JEPA visual analysis MUST 在报告中单独总结 GPS shortcut reliance 相关发现，包括 drop GPS、misleading GPS、temporal delay、GPS-only collapse slope、JEPA 与 GPS-centric baseline 的 clean-delta 和 caveat。报告 MUST 明确区分性能结果、反事实 intervention 和解释性诊断。

#### Scenario: 报告 GPS shortcut 结论
- **WHEN** benchmark 产物包含 drop GPS 或 misleading GPS 条件
- **THEN** `report.md` MUST 包含 GPS shortcut reliance 小节
- **AND** 小节 MUST 引用对应表格或图表路径
- **AND** 小节 MUST 标记哪些结论来自 counterfactual intervention

#### Scenario: 报告避免过度声称
- **WHEN** attention、gradient 或 ablation reliance 诊断被用于解释模型行为
- **THEN** `report.md` MUST 将其标记为解释性证据
- **AND** 报告 MUST 不把 attention 或 gradient 单独描述为因果证明

### Requirement: Benchmark case study 选择
JEPA visual analysis MUST 支持根据 benchmark 条件选择 case study，至少覆盖 JEPA 在 GPS collapse 下优于 GPS-centric baseline 的 `jepa_recovery`、GPS-centric baseline 在 clean GPS 下占优但在 distractor 下失败的 `gps_shortcut_failure`、以及所有模型失败的 `shared_failure`。

#### Scenario: 选择 JEPA recovery case
- **WHEN** comparison table 中存在 JEPA 模型在 GPS collapse 条件下保持 Top-K hit 而 GPS-centric baseline 失败的样本
- **THEN** 分析流程 MUST 按 deterministic seed 和排序规则选择 case
- **AND** 系统 MUST 写出 case selection 表和机器可读 case payload

#### Scenario: 包含 shared failure
- **WHEN** benchmark 条件中存在所有模型均远错或指标显著下降的样本
- **THEN** 分析流程 MUST 可选择 `shared_failure` case
- **AND** report MUST 将其标记为失败模式，而不是只展示成功案例

### Requirement: Benchmark 分析降级与可测试性
JEPA visual analysis MUST 为 benchmark manifest ingestion、robustness matrix 生成、shortcut report 和缺失可选诊断的降级行为提供测试。缺少 optional visualization dependency 或样本数不足时，分析流程 MUST 记录 warning 并尽可能输出已有表格。

#### Scenario: 缺少可视化依赖
- **WHEN** matplotlib、UMAP、Grad-CAM 或其它 optional visualization dependency 不可用
- **THEN** 分析流程 MUST 写出可生成的 CSV/JSON 表格
- **AND** manifest 和 report MUST 记录图表跳过原因
- **AND** CLI MUST 成功完成，除非必需的 benchmark 指标输入不可用

#### Scenario: mock benchmark manifest 测试
- **WHEN** 单元测试使用 mock benchmark manifest 和小型 metrics 表运行分析 helper
- **THEN** 系统 MUST 生成 analysis manifest、robustness summary 表和 report skeleton
- **AND** 输出文件清单中的生成项 MUST 存在或被标记为 skipped

