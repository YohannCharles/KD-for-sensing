## Context

`kd-sensing-jepa-visual-analysis` 已经能导出 GPS-query attention 的 patch-grid 图、query/time 分面图和 attention summary 表，但这些图没有映射到图像内容上，用户无法直观看出高权重 patch 对应道路、车辆、建筑或天空。现有 `DeepSense6GDataset` metadata 会记录 `image_path`，因此可以把 14x14 attention heatmap 直接缩放回原始 RGB 图片大小，再按论文图常见的低对比底图 + 彩色热力图方式叠加；forward batch 中的模型输入 RGB tensor 只作为 raw image 读取失败时的兜底。

参考依据很直接：Grad-CAM 类可视化通常把低分辨率定位热图上采样后叠加到图像上；Attention Rollout 提醒 raw attention 只是 post hoc 视角，token 间信息会跨层混合；Chefer 等 Transformer 解释工作也指出简单 attention/平均 attention 容易模糊信号。因此本 change 只把当前 GPS-query attention 标注为诊断可视化，不把它包装成因果归因。

参考论文：
- Selvaraju et al., Grad-CAM, ICCV 2017: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf
- Abnar and Zuidema, Quantifying Attention Flow in Transformers, 2020: https://arxiv.org/abs/2005.00928
- Chefer et al., Transformer Interpretability Beyond Attention Visualization, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf

## Goals / Non-Goals

**Goals:**
- 为提供 attention diagnostics 的 GPS-query 模型导出论文式 heatmap overlay，效果接近示例图中的蓝/绿/黄/红热力叠加。
- overlay 优先使用 metadata 中的原始 RGB 图片尺寸：把 14x14 heatmap 插值到 raw image height/width 后叠加。
- 保留 query/time 维度，至少输出每个 selected sample 的 query 面板。
- 在 `attention_summary.csv`、`analysis_manifest.json` 或 report 中记录 overlay 是否生成、归一化方式和 caveat。
- 缺少 image 或 attention 时安全跳过 overlay，已有表格和 patch-grid 图继续生成。

**Non-Goals:**
- 不新增模型架构或训练损失。
- 不实现 Grad-CAM、gradient rollout、LRP 或 ablation attribution。
- 不声称 overlay 是 causal explanation。
- 不新增依赖，也不读取或修改 checkpoint、训练日志、split CSV。

## Decisions

1. 使用原始 RGB 图片作为 overlay 底图。
   - 原因：用户需要可读的原图叠加图，而不是只能看到模型 resize 后的小图；14x14 heatmap 直接按原图宽高插值，最接近论文图示的呈现方式。
   - 兜底：若 `metadata.image_path` 不存在、文件不可读或样本来自 mock batch，则从模型输入 tensor 反归一化得到 RGB 图并标记 `overlay_image_source=model_input_tensor`。

2. 继续沿用现有 attention 收集路径。
   - 在 `_collect_forward_outputs` 中，当 `attention_detail_maps` 保存某个 sample 时，同时保存对应的 metadata row 和可选 model-input RGB 兜底图。
   - 不新增单独缓存格式，最多给 `ModelAnalysis` 增加 `attention_metadata` 和 `attention_images` 字段。

3. overlay 使用固定默认样式 `paper_overlay`。
   - attention grid 用双线性插值优先上采样到 raw image height/width。
   - 每个样本 query/time 面板共享归一化范围。
   - 底图做低对比、低饱和、轻冷色处理，保留场景结构；热图使用类似 `jet`/`turbo` 的蓝-绿-黄-红 colormap 以固定 alpha 叠加。
   - 图题和 CSV/report 明确写 `normalization=per_sample_shared_minmax` 与 `overlay_style=paper_overlay`。

4. 保留旧图。
   - `figures/attention_cases/` 和 `figures/attention_query_time_cases/` 不删除。
   - 新增 `figures/attention_image_overlays/`，避免破坏旧报告路径。

## Risks / Trade-offs

- attention map 本身不稳定或 query 间差异很弱 -> report 必须写 caveat，不能把亮区解释成因果证据。
- raw image 与模型输入 resize 后的 token 空间可能存在尺度/长宽比差异 -> 图注必须说明 `overlay_image_source=raw_image`，并记录这是用于阅读的 post hoc attention overlay。
- 样本多时 PNG 数量增加 -> 复用 `max_attention_cases` 限制，不新增全量导出。
- 缺少 image modality 或 mock batch image -> 记录 `attention_image_overlay_unavailable`，不影响 CLI 成功结束。

## Migration Plan

实现只新增诊断输出，不需要迁移已有训练或分析产物。已有分析命令重跑后会在同一 output dir 下多出 `figures/attention_image_overlays/`；回滚时删除新增 helper/字段即可，旧 attention 图和 summary 表不受影响。

## Open Questions

无必须阻塞实现的问题。后续若需要论文级解释性证据，再单独评估 gradient rollout、Grad-CAM 或 ablation 方案。
