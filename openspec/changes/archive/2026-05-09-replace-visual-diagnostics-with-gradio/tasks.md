## 1. 目录与依赖

- [x] 1.1 新增 `tools/visualization/` 目录结构，包含 Gradio viewer、viewer utils、README、示例 manifest 和 viewer 专用 requirements。
- [x] 1.2 在 `tools/visualization/requirements_viewer.txt` 或 `pyproject.toml` optional extra 中记录 `gradio`、`plotly`，并保留 `numpy`、`pandas`、`pillow` 复用说明。
- [x] 1.3 使用 `conda run -n kd_mm_beam python -c "import gradio, plotly"` 或等价命令确认交互可视化依赖安装状态，未安装时在 README 中给出安装命令。

## 2. Manifest 工具函数

- [x] 2.1 在 `viewer_utils.py` 实现 `load_manifest()`，支持 JSON 数组和 JSONL，并为样本补充稳定内部 index。
- [x] 2.2 实现 `resolve_path()`、`safe_get()`、`load_image_safe()` 和 `load_json_safe()`，确保缺失路径、坏图片和坏 JSON 不会抛出未处理异常。
- [x] 2.3 实现 scene/split 枚举与 `filter_samples()`，支持 `all`、`correct only`、`wrong only`、`low quality only`。
- [x] 2.4 实现 `make_empty_figure()`、`make_gps_figure()`、`make_mmwave_figure()`、`make_score_bar()` 和 `dict_to_dataframe()`。
- [x] 2.5 实现 `build_info()`，只返回当前样本最关键的元数据、label、prediction 和 extra。

## 3. Gradio Viewer

- [x] 3.1 在 `gradio_multimodal_viewer.py` 实现 argparse 参数：`--config`、`--manifest`、`--cache-dir`、`--force-rebuild`、`--sample-limit`、`--host`、`--port`、`--share`、`--debug`。
- [x] 3.2 使用 `gr.Blocks` 构建顶部控制区：scene、split、show mode、slider、样本编号、上一帧、下一帧、自动播放、播放速度和 timer。
- [x] 3.3 构建 Overview、Raw Modalities、Processed Modalities、Diagnostics 分区或 Tabs，展示 raw/processed 五模态和诊断 JSON/图表/表格。
- [x] 3.4 实现 `render_sample()`，按当前过滤结果加载当前样本并返回固定顺序的所有 Gradio 输出。
- [x] 3.5 实现 filter change、slider change、prev/next click、timer tick 事件绑定，确保 slider 范围和页面输出同步更新。
- [x] 3.6 处理空 manifest、空过滤结果和所有模态缺失场景，页面显示 `No samples found` 或 Missing / Not Available。

## 4. Manifest 导出与旧入口迁移

- [x] 4.1 新增或改造 `tools/visualization/export_viewer_manifest.py`，从现有训练配置、Dataset 元数据、标签和可选预测/quality/gate 文件处理全部样本并导出 viewer cache/manifest。
- [x] 4.2 导出脚本必须遵循现有 cache policy，并在 `read_only` 或 `off` 时不写入新的训练 cache 文件；viewer cache 需记录 metadata 并在源文件未变化时复用。
- [x] 4.3 退役 `src/kd_sensing/diagnostics/visualization/*` 的静态 PNG 主流程，或将旧 CLI 改为 manifest 导出/迁移提示。
- [x] 4.4 更新或移除 `src/kd_sensing/cli/visualize_modalities.py`、`scripts/visualize_modalities.py`、`configs/diagnostics/modality_visualization.yaml` 和 `pyproject.toml` 中旧 console script 的相关引用。

## 5. 文档与示例

- [x] 5.1 编写 `tools/visualization/README.md`，说明功能、依赖安装、manifest 格式、运行命令、常见问题和迁移路径。
- [x] 5.2 新增 `tools/visualization/sample_manifest_example.json`，覆盖 raw、processed、label、prediction、confidence、quality、gate 和缺失字段示例。
- [x] 5.3 README 中所有 Python 命令示例必须使用 `conda run -n kd_mm_beam` 或明确说明先激活 `kd_mm_beam` 环境。

## 6. 测试与验证

- [x] 6.1 用单元测试覆盖 manifest 读取、路径解析、过滤逻辑、GPS/mmWave 图表、score table 和缺失字段容错。
- [x] 6.2 更新 `tests/test_modality_visual_diagnostics.py`，删除静态 PNG/summary 主产物断言，改为验证 manifest 导出或旧入口迁移提示。
- [x] 6.3 新增 viewer CLI smoke 测试，使用示例 manifest 和 `--config --check-only` 验证 `gradio_multimodal_viewer.py` 可以构建 Blocks、准备 cache，且不启动训练或加载 checkpoint。
- [x] 6.4 运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 验证迁移后的可视化相关测试。
- [x] 6.5 运行 `conda run -n kd_mm_beam pytest` 或与本变更相关的完整测试子集，确认旧静态入口退役没有破坏训练、数据加载和评估路径。

## 7. Raw/Processed 语义纠偏

- [x] 7.1 修改 manifest 导出：LiDAR raw 从点云源文件生成独立俯视预览，processed LiDAR 继续来自 Dataset BEV，且两者不能复用同一张量。
- [x] 7.2 修改 manifest 导出：Radar raw/processed 写入数据空间 metadata，当前 RA/DA 序列路径标注为 precomputed 而不是原始 radar cube。
- [x] 7.3 修改 processed GPS JSON 与 viewer 图表：relative-polar/标准化 GPS 特征按 time-index 多曲线展示，不再当二维轨迹画。
- [x] 7.4 修改 processed mmWave JSON 与 viewer 图表：记录并展示 dB 或标准化数值空间。
- [x] 7.5 bump viewer manifest cache version，避免复用旧的错误 raw LiDAR cache。
- [x] 7.6 增加测试覆盖 LiDAR raw/processed 差异、processed GPS 特征图、mmWave scale metadata 和 Radar space metadata。
- [x] 7.7 运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py`。

## 8. Future Beam Distribution Inspector

- [x] 8.1 在 `viewer_utils.py` 新增 future horizon、beam distribution 提取、rank、entropy、top1 margin、summary/detail 和 Plotly heatmap/per-modality 纯函数。
- [x] 8.2 在 Gradio Diagnostics Tab 新增 Future Beam Distribution Inspector 控件、plot、summary dataframe 和 detail JSON，并把 sample/filter/nav/timer 事件全部接入新输出。
- [x] 8.3 扩展模型预测导出和 manifest 合并，让 `beam_distribution[modality].prob/logit` 可以随 `--run-models` 写入 viewer manifest。
- [x] 8.4 更新 README 和示例说明，记录 `beam_distribution` manifest 结构、probability/logit 语义和旧字段兼容限制。
- [x] 8.5 增加测试覆盖 probability/logit 分布、horizon fallback、summary 指标、Gradio render 输出和预测导出字段。
- [x] 8.6 运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py`。
