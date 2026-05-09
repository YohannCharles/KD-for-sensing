## Why

当前 Gradio 多模态 viewer 在切换下一帧和跨 20 帧跳转时仍有明显延迟，用户观测到下一帧约 `0.5s + 1.1s`、切换 20 帧约 `2.0s + 1.7s`。这会影响连续排查时序样本的效率，尤其是在 scene/split 过滤后需要快速比较 raw、processed 和 diagnostics 的场景。

## What Changes

- 优化 `replace-visual-diagnostics-with-gradio` 引入的 Gradio viewer 切帧路径，不调整页面布局、Tab 结构、控件位置或展示内容。
- 为 scene/split/show mode 建立过滤结果缓存，避免每次 slider、上一帧、下一帧、timer tick 都重新扫描完整 manifest。
- 将图片输出改为优先返回可解析的文件路径，减少每次回调中重复 `PIL.Image.open().convert().copy()` 和图片二次序列化成本。
- 将样本渲染缓存拆分为模态图片、Plotly 图表、DataFrame/JSON 和 future distribution 控件相关结果，避免 horizon/chart 改变时重算与其无关的 raw/processed 模态。
- 对隐藏 Tab 采用惰性同步：切帧时优先更新当前可见内容和必要状态，用户切换 Tab 时再刷新该 Tab 的当前样本内容；页面布局保持不变。
- 增加轻量性能诊断输出或测试辅助函数，能区分过滤耗时、后端渲染耗时、序列化/返回输出数量等因素，帮助判断优化是否有效。
- 明确性能边界：Gradio 前端对大量 image/Plotly/DataFrame 组件的渲染仍有不可完全消除的固定成本；本变更目标是减少后端计算和同步输出体积，而不是保证所有机器上达到固定毫秒级延迟。

## Capabilities

### New Capabilities
- `gradio-viewer-performance`: 定义 Gradio viewer 在样本切换、过滤、自动播放和 Tab 同步时的性能与可观测性要求。

### Modified Capabilities
（无）

## Impact

- 受影响代码：
  - `tools/visualization/gradio_multimodal_viewer.py`
  - `tools/visualization/viewer_utils.py`
  - `tests/test_modality_visual_diagnostics.py`
  - 可选更新 `tools/visualization/README.md` 中的性能说明或调试开关。
- API / CLI 影响：
  - 现有启动参数、manifest 格式和页面布局保持兼容。
  - 如新增性能调试参数，应默认关闭，不影响普通浏览。
- 行为影响：
  - 切帧、上一帧、下一帧和自动播放应减少重复过滤、重复图片解码和隐藏 Tab 的同步输出负担。
  - 隐藏 Tab 内容允许在 Tab 被选中时刷新到当前样本，但用户可见区域不得显示旧样本。
  - Viewer 仍保持只读，不修改训练 checkpoint、训练日志、评估报告或 split CSV。
