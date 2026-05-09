## Why

现有可视化以静态 PNG、JSON 摘要和离线报告为主，适合留档但不适合快速浏览多模态时序样本、定位错误预测、比较 raw/processed 表示和检查 confidence/quality/gate 等诊断信号。当前项目已经有 image、radar、LiDAR、GPS、mmWave、融合模型、CRAF/MARF 等多模态组件，下一步更需要一个交互式数据分析入口，而不是继续堆叠静态可视化产物。

## What Changes

- 新增基于 Gradio 的交互式多模态数据分析页面，支持选择数据集配置、自动处理全部样本、按时间轴浏览样本、上一帧/下一帧、自动播放、scene/split/show mode 过滤和缺失字段容错。
- 新增可复用 viewer cache。Gradio viewer 可通过 `--config` 自行调用 Dataset 生成 `samples.json`、`manifest_meta.json` 和 processed assets；配置、CSV、样本源文件或资产未变化时直接复用缓存。
- 保留 manifest 读取能力作为离线/复用入口，但不再要求用户先手动生成 manifest。
- 支持同屏展示 raw 与 processed 的 image、LiDAR、radar、GPS、mmWave，以及 label、prediction、top-k、confidence、quality、gate、extra 等诊断信息。
- 新增 Future Beam Distribution Inspector，支持按 `t+1...t+H` horizon 查看 image、LiDAR、radar、GPS、mmWave 和可选 fusion 在所有 beam label 上的完整 probability/logit 分布，并输出 GT rank、entropy、top1 margin、distance_to_gt 等摘要指标。
- 扩展 viewer manifest 的预测诊断格式，支持 `beam_distribution[modality].prob/logit`；旧的 `modality_prediction` / per-modality top1 摘要只能用于表格，不用 top1 confidence 伪造完整分布。
- 修正 raw/processed 语义：LiDAR raw 必须来自点云源文件的独立预览而不是复用 Dataset BEV；GPS processed 必须按特征时间序列展示；mmWave processed 必须标明 dB 或标准化空间；Radar raw 若来自预生成 RA/DA，界面和 metadata 必须明确它不是原始 radar cube。
- 新增 manifest 导出计划，将现有 Dataset 元数据、处理后张量路径、标签、预测结果、质量分数和 gate 权重合并为 viewer 可读取的 `samples.json` 或 JSONL。
- **BREAKING**：不再以现有静态 PNG 总览图和 `summary.json` 报告作为主可视化方案；旧的 `kd-sensing-visualize-modalities` / `scripts/visualize_modalities.py` 路径应被替换、退役或改为 manifest 导出辅助能力，避免维护两套重复可视化逻辑。
- 新增独立可选依赖说明，Gradio、Plotly 等交互可视化依赖不强制进入训练主依赖，除非项目决定提供统一 extra。

## Capabilities

### New Capabilities

- `gradio-visual-analysis`: 定义 Gradio 交互式多模态样本浏览、过滤、自动播放、raw/processed 展示、诊断指标展示、manifest 读取与容错行为。

### Modified Capabilities

- `modality-visual-diagnostics`: 将现有静态文件诊断能力收敛为 Gradio viewer 的 manifest 数据准备与兼容辅助能力，不再要求生成静态 PNG 总览图作为主验收产物。

## Impact

- 受影响代码：
  - 新增 `tools/visualization/gradio_multimodal_viewer.py`
  - 新增 `tools/visualization/viewer_utils.py`
  - 新增 `tools/visualization/README.md`
  - 可选新增 `tools/visualization/export_viewer_manifest.py`
  - 可选新增 `tools/visualization/sample_manifest_example.json`
  - 可选新增 `tools/visualization/requirements_viewer.txt`
- 受影响现有入口：
  - `src/kd_sensing/diagnostics/visualization/*`
  - `src/kd_sensing/cli/visualize_modalities.py`
  - `scripts/visualize_modalities.py`
  - `pyproject.toml` 中 `kd-sensing-visualize-modalities` console script
  - `configs/diagnostics/modality_visualization.yaml`
  - `tests/test_modality_visual_diagnostics.py`
- 依赖影响：
  - 新增交互可视化依赖 `gradio`、`plotly`
  - 复用现有 `numpy`、`pandas`、`pillow`
  - 依赖应优先放入可选安装说明或专用 requirements，避免训练环境默认变重。
- 行为影响：
- Viewer 默认只读训练产物，只在 viewer cache 目录写入 `samples.json`、metadata 和 processed assets，不修改训练 checkpoint、训练日志、评估报告或 split CSV。
  - Viewer 对缺失文件、缺失模态、坏 JSON 和空过滤结果必须降级显示，不得使页面崩溃。
  - 已生成的旧 viewer cache 中存在 raw LiDAR 与 processed LiDAR 完全相同的语义错误；实现必须提升 manifest cache version，确保用户重建后不继续复用旧资产。
  - 旧静态可视化的测试与规范需要更新为 Gradio viewer 与 manifest 导出的验收标准。
