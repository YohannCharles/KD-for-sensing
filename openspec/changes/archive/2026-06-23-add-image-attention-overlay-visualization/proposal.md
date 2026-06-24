## Why

当前 GPS-query attention 图只显示 14x14 patch 热力格，用户很难判断模型到底看到了原图中的道路、车辆、建筑还是天空区域。需要把 attention 映射回输入图像，并按论文常见做法将原图压暗后叠加热力图，产出可直接阅读和放进报告的样本级可视化。

## What Changes

- 为 JEPA visual analysis 增加 image-space attention overlay：把 `[time, query, patch]` attention reshape 到 patch grid、上采样到原图尺寸，并以论文图常见的低对比底图 + 彩色热力图方式叠加。
- 输出 query/time 级别 overlay 面板，保留当前 patch-grid attention 图和 summary CSV。
- overlay 图必须标注模型名、样本索引或 sample id、target beam、Top-k 预测、history frame index、query index 和 attention 归一化方式。
- 当原图或 attention 不可用时安全降级：跳过 overlay，继续写出已有 attention summary 和 manifest/report warning。
- 不改变训练、checkpoint、正式评估指标或模型结构。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `jepa-visual-analysis-suite`: GPS-query attention 可视化从纯 patch-grid 图增强为可选 image-space overlay，并明确输出、降级和报告 caveat。

## Impact

- 主要影响 `src/kd_sensing/diagnostics/jepa_visual_analysis.py` 和对应测试。
- 继续使用 `kd-sensing-jepa-visual-analysis` 入口；不新增 CLI。
- 输出新增到 ignored runtime 目录下的 `figures/attention_image_overlays/`，并登记到 `analysis_manifest.json`、`report.md` 或 HTML gallery。
- 使用现有图像读取、matplotlib/Pillow/torchvision 能力；不为叠图新增依赖。
