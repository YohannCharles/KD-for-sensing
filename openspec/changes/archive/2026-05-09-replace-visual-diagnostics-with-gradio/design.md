## Context

当前项目已经具备多模态数据加载、预处理、训练、评估和静态诊断能力。现有 `modality-visual-diagnostics` 入口通过训练配置构建 Dataset，抽样后生成 PNG 总览图、`samples.jsonl`、`samples.csv`、`split_stats.json` 和 `summary.json`。这套方式能留存证据，但每次查看新样本或切换过滤条件都需要重新运行脚本，且 raw/processed、预测、confidence、quality、gate 等信息分散在静态产物里，不适合排查 scene32 这类序列域偏移和模态失衡问题。

新方案以 Gradio 为主可视化入口，采用“配置驱动处理 + manifest cache 展示”。Gradio viewer 支持直接传入训练/数据集配置，启动前自行构建 Dataset、处理配置中所选 split 的全部样本，并写入 `samples.json`、`manifest_meta.json` 和 processed assets。再次运行时先校验配置、CSV、样本源文件和已生成资产，未变化则直接复用 cache。离线 `samples.json` 或 JSONL 读取能力继续保留，便于调试和复用。

## Goals / Non-Goals

**Goals:**

- 提供一个可本地启动的 Gradio 多模态时序样本浏览器。
- 支持通过 `--config` 选择数据集配置，自动处理全部样本并缓存。
- 支持 image、LiDAR、radar、GPS、mmWave 的 raw 与 processed 同屏查看。
- 支持 label、prediction、top-k、correct、confidence、quality、gate、extra 等诊断信息展示。
- 支持 scene、split、show mode、slider、prev/next、自动播放等交互。
- 对缺失字段、缺失文件、坏 JSON 和空过滤结果做安全降级。
- 将旧静态 PNG 诊断入口退役或改为 manifest/export 辅助能力，减少重复可视化逻辑。
- 以可选依赖方式引入 Gradio 和 Plotly，训练主流程不因可视化功能变重。

**Non-Goals:**

- 第一版不在 Gradio 页面内直接运行模型推理或训练。
- 第一版不实现在线编辑样本标签、保存人工备注或导出页面截图。
- 第一版不追求复杂前端定制，不引入 React/FastAPI 服务。
- 第一版不保证旧 `summary.json` / PNG 报告格式兼容；旧方案不作为保留路径。

## Decisions

### Decision 1: 使用 config 驱动的数据准备和 manifest cache 展示

Gradio viewer 的主入口为 `--config`。程序先根据配置构建 Dataset，处理所选 split 的全部样本，生成 `samples.json` 或 JSONL 等价结构；页面展示阶段再读取该 manifest。每条样本记录包含样本元数据、raw/processed 路径、标签、预测、confidence、quality、gate 和 extra 信息。相对路径按 manifest 所在目录优先解析，也允许通过项目根目录兜底解析。

理由：
- 用户只需要选择数据集/训练配置，不需要先手动准备 manifest。
- 页面逻辑可以围绕“当前样本记录”实现，测试更简单。
- 模型输出、质量分数和 gate 权重来源可能不同，manifest 是稳定的汇合点。
- 生成后的 manifest/cache 可复用，后续启动快。

备选方案是只允许手动导出 manifest。该方案隔离性强，但不符合“选择数据集后程序自行处理”的目标。

### Decision 2: 使用 metadata fingerprint 复用处理结果

每个 viewer cache 写入 `manifest_meta.json`，记录配置摘要、样本源文件路径、mtime/size、样本数和 cache digest。启动时如果 metadata、manifest、processed assets 和源文件状态都一致，则复用 cache；配置或源文件变化、资产缺失、传入 `--force-rebuild` 时重处理。

### Decision 3: 拆分 viewer 与 export

新增 `tools/visualization/gradio_multimodal_viewer.py` 作为 UI 入口，`tools/visualization/viewer_utils.py` 放置纯函数工具；可选新增 `tools/visualization/export_viewer_manifest.py` 将现有 Dataset 元数据、处理后样本路径和预测诊断数据导出为 viewer manifest。

理由：
- Viewer 可以在没有完整 DeepSense6G 数据集的环境中用示例 manifest 做 smoke test。
- Export 脚本可以复用项目 Dataset 与评估输出，并遵循 `conda run -n kd_mm_beam` 环境。
- 后续若预测结果格式变化，只需要改 export 或 manifest 合并逻辑，不影响 UI 核心浏览逻辑。

### Decision 4: 退役旧静态主入口

不再要求旧 `kd-sensing-visualize-modalities` 生成 PNG 总览和静态 summary 作为主可视化产物。实现时可以删除旧脚本与配置，也可以短期保留一个兼容入口，但该入口应打印迁移提示或调用 manifest 导出，不应继续维护单独的静态渲染逻辑。

理由：
- 用户明确要求不用保留现有可视化方案，减少冗余。
- 现有静态入口已有大量采样、统计、渲染、文件命名逻辑，继续并行维护会增加测试与文档成本。
- Gradio viewer 覆盖了样本浏览与诊断分析的主要需求，静态 PNG 不是当前最有效的分析形态。

### Decision 5: 使用 Plotly 作为 Gradio 图表对象

GPS、mmWave、confidence、quality、gate 等数值展示统一返回 Plotly Figure。图片模态返回 PIL Image 或 `None`；mmWave 若源文件为图片，可转为 Plotly image figure 或使用单独 image 组件展示。

理由：
- Plotly 与 `gr.Plot` 配合稳定，支持 bar、line、heatmap 和空状态。
- 统一图表返回类型后，缺失数据和错误数据可以通过 `make_empty_figure()` 表示。

### Decision 6: 依赖放在可选安装路径

`gradio`、`plotly` 写入 `tools/visualization/requirements_viewer.txt`，并可在 `pyproject.toml` 中增加 `visualization` optional extra。训练核心 dependencies 不强制新增这些包。

理由：
- 项目训练与评估不依赖 Web UI。
- 交互可视化包体积和版本变化较大，隔离依赖能降低训练环境风险。

### Decision 7: raw/processed 必须表达不同处理阶段

Manifest 导出时把“源数据预览”和“Dataset 处理后张量”分开生成：

- LiDAR raw 从最后一个历史点云源文件读取点云，按原始点位置生成俯视预览；processed LiDAR 继续保存 Dataset 返回的 BEV 张量。
- Radar raw 在当前序列 CSV 指向 RA/DA 文件时命名为 precomputed radar preview，并在 manifest metadata 里记录 `space: precomputed_ra_da`；processed radar 记录 Dataset 裁剪/加载后的 `space: dataset_ra_da`。不假装它是原始 radar cube。
- GPS raw 使用经纬度轨迹；processed GPS 写出 `{features, feature_names, feature_space}`，viewer 用多条时间序列展示 relative-polar 或标准化后的特征。
- mmWave raw 使用 power vector；processed mmWave 写出 `{beam_power_seq, scale, normalized}`，viewer 标明 dB 或 z-score 空间。

理由：
- 用户排查的是预处理是否生效，raw/processed 如果画同一个张量会直接误导。
- GPS relative-polar 特征不是坐标，不能用轨迹图解释。
- mmWave dB 压缩会保持峰值形状，必须通过标题/metadata 说明数值空间变化。
- Radar 的序列 CSV 已经使用 RA/DA 预处理产物，第一版不强行反向映射原始 cube，避免引入脆弱路径推断。

### Decision 8: Future beam 分布默认展示 probability，保留 logit 调试入口

Diagnostics Tab 增加 Future Beam Distribution Inspector。用户选择 `t+1...t+H` horizon 后，viewer 从 `sample["beam_distribution"][modality]["prob"|"logit"][horizon_index]` 读取完整 beam 分布，支持 heatmap 总览、per-modality subplot 和摘要表格。默认 view type 是 softmax 后 probability，因为概率范围固定为 0 到 1、每个 horizon 总和为 1，GT confidence、GT rank、entropy 和 top1 margin 更容易解释；logit 模式只作为高级调试入口，用于检查不同模态 head 的 logit scale。

该模块不假设 beam 数固定为 64，而是从分布长度推断。Heatmap 要求参与展示的模态 beam 数一致；不同长度的异常模态会被跳过，并在 detail JSON 中记录 warning。若当前样本没有 `beam_distribution`，图形显示空状态；若只存在旧的 `modality_prediction` / `prediction.modalities` 摘要字段，viewer 只生成 summary/detail，不用 top1 confidence 伪造完整分布。

模型预测导出在已有 `confidence_curves` 的基础上同时写入：

```json
"beam_distribution": {
  "gps": {
    "prob": [[...]],
    "logit": [[...]]
  }
}
```

Manifest 合并逻辑负责把各单模态预测结果合并到样本顶层的 `beam_distribution`，viewer UI 只读取 manifest，不在页面回调里重复运行模型。

## Risks / Trade-offs

- Manifest 与真实 Dataset 不一致 → 导出脚本必须记录来源路径、split、dataset index、label、处理后路径和导出时间；README 明确 viewer 只信任 manifest。
- 大 manifest 启动慢 → 第一版一次性读取元数据但不预加载图片；后续可增加 JSONL 流式索引或分页。
- Gradio 事件返回值较多，维护成本高 → 将输出组件列表集中定义，`render_sample()` 返回固定顺序；工具函数保持无 UI 副作用。
- mmWave/GPS 输入格式不统一 → 工具函数支持常见 dict/list/path 格式，无法识别时返回空图而不是报错。
- LiDAR raw 点云预览比 BEV 预览更稀疏 → 这是期望行为；processed BEV 才承担训练输入可视化。
- 旧测试大量覆盖静态产物 → 任务中必须删除或改写旧测试，新增 viewer utils、manifest 读取、过滤、容错和 CLI smoke 测试。
- 退役旧入口是破坏性变更 → README、proposal 和 CLI 文案必须明确迁移路径：先导出 manifest，再启动 Gradio viewer。

## Migration Plan

1. 新增 Gradio viewer、viewer utils、README、示例 manifest 和 viewer 专用 requirements。
2. 新增或改造 manifest 导出脚本，优先支持从现有 Dataset/CSV/评估输出合并基础样本记录。
3. 移除或退役旧静态可视化代码、配置、CLI 和 console script；若短期保留，入口只做迁移提示或调用 export。
4. 将 `tests/test_modality_visual_diagnostics.py` 改为覆盖 Gradio manifest 工具与导出行为，删除 PNG/summary 作为主产物的断言。
5. 使用 `conda run -n kd_mm_beam pytest ...` 验证相关测试；手工用示例 manifest 启动 `conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py --manifest ... --port 7860` 做 smoke check。

Rollback 策略：如果 Gradio 入口无法在目标环境运行，可暂时恢复旧静态诊断入口的最近归档版本；但不建议长期双轨维护。
