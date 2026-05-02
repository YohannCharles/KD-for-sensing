## 1. 配置与采样

- [x] 1.1 扩展 `VisualizationConfig` 和配置解析，支持 `per_seq_sample_count` 与 `compare_scenes`，并保持旧配置默认兼容。
- [x] 1.2 实现按 `seq_index` 分层采样，输出每个序列的候选数、请求数、实际数和选中 dataset index。
- [x] 1.3 更新 `configs/diagnostics/modality_visualization.yaml`，加入新的诊断字段示例。

## 2. 聚合统计与跨场景输出

- [x] 2.1 新增 split/`seq_index` 聚合统计计算，汇总 image mask density、radar RA/DA std、LiDAR nonzero fraction 和 future label 分布。
- [x] 2.2 写出 `split_stats.json`，并在 `summary.json`、返回值和输出文件列表中记录路径。
- [x] 2.3 实现 `compare_scenes` 多 scene 运行封装，确保每个 scene 使用独立输出子目录并生成总摘要。

## 3. 样本图布局

- [x] 3.1 调整单样本 PNG 布局，使用更大的 figure、`constrained_layout`、短标题和按模态分区的行结构。
- [x] 3.2 增强 image 面板，启用 raw preview 时展示 raw reference 与 processed motion mask 对照，并在样本记录中标记 reference。
- [x] 3.3 增强 radar/LiDAR 面板，radar RA/DA 使用共享色标，LiDAR 展示总体与通道级非零统计摘要。

## 4. 验证

- [x] 4.1 更新 `tests/test_modality_visual_diagnostics.py`，覆盖分层采样、`split_stats.json`、summary 路径和新配置兼容性。
- [x] 4.2 使用 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 运行目标测试。
- [x] 4.3 使用 `conda run -n kd_mm_beam openspec status --change "enhance-modality-visualization-diagnostics"` 检查 OpenSpec 状态。

## 5. 元数据非覆盖输出

- [x] 5.1 扩展诊断配置，新增 `preserve_existing_outputs`，默认保留已有元数据产物。
- [x] 5.2 实现同批次非冲突元数据路径解析，覆盖 `summary`、`samples.jsonl`、`samples.csv`、`split_stats` 和 `final_config`。
- [x] 5.3 更新单 scene 与 `compare_scenes` 输出逻辑，返回值、summary 和 `output_files` MUST 使用实际非覆盖路径。
- [x] 5.4 更新 `tests/test_modality_visual_diagnostics.py`，覆盖重复运行不覆盖已有 JSON/JSONL/CSV/YAML 文件。
- [x] 5.5 使用 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 运行目标测试。
- [x] 5.6 使用 `conda run -n kd_mm_beam openspec instructions apply --change "enhance-modality-visualization-diagnostics" --json` 确认所有任务完成。
