## MODIFIED Requirements

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
