## 1. Attention 底图收集

- [x] 1.1 在 `ModelAnalysis` 中增加受 `max_attention_cases` 限制的 overlay metadata/source image 存储字段。
- [x] 1.2 在 `_collect_forward_outputs` 中为已保存 attention detail map 的样本同步保存 metadata row，并保留模型输入 RGB frame 作为原图缺失时的兜底。
- [x] 1.3 增加原始 RGB 图片解析 helper，优先从 `metadata.image_path` 读取原图；失败时使用 image tensor 反归一化兜底，二者都缺失时返回 skipped reason。

## 2. Overlay 图生成

- [x] 2.1 增加 attention grid 上采样到原始图片尺寸的 helper，使用同一样本共享归一化范围。
- [x] 2.2 增加 `paper_overlay` 绘图 helper：低对比底图 + 蓝/绿/黄/红 heatmap 半透明叠加，输出 query/time 面板。
- [x] 2.3 在 `_write_attention_outputs` 中写出 `figures/attention_image_overlays/`，保留现有 `attention_cases/` 和 `attention_query_time_cases/`。
- [x] 2.4 在 manifest、report 或 warnings 中记录 overlay 产物、归一化方式和 `attention_image_overlay_unavailable` 降级原因。

## 3. 验证与目标产物

- [x] 3.1 为 synthetic attention/raw image 增加 focused test，检查 overlay 文件按原图尺寸写出且缺 image 时安全跳过。
- [x] 3.2 运行 `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q`。
- [x] 3.3 用 `pooler_gps_query_k2_tokens` 的现有分析配置重跑用户关心的可视化，并确认新增 overlay 图在目标 output dir 下可打开。
