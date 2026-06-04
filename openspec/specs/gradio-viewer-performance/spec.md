# gradio-viewer-performance Specification

## Purpose
Define frame-navigation performance requirements for the Gradio multimodal viewer, including filtered-index caching, path-based image outputs, split render caches, lazy tab synchronization, and optional profiling logs, so interactive diagnostics remain responsive without changing the viewer layout or data contract.
## Requirements
### Requirement: 切帧路径缓存过滤结果
Gradio viewer MUST cache filtered sample indices for each `(scene, split, show_mode)` combination during a viewer session. Slider changes, previous/next clicks, and timer ticks MUST reuse the cached filtered indices when the filter controls have not changed.

#### Scenario: 下一帧复用过滤结果
- **WHEN** 用户在 scene、split 和 show mode 不变的情况下点击下一帧
- **THEN** viewer MUST NOT rescan the entire manifest to compute the same filtered sample list
- **AND** viewer MUST compute the next index from the cached filtered indices

#### Scenario: 过滤条件变化后刷新缓存
- **WHEN** 用户修改 scene、split 或 show mode
- **THEN** viewer MUST compute the filtered sample indices for the new combination
- **AND** subsequent navigation events MUST reuse the new cached result

### Requirement: 图片输出避免重复解码
Gradio viewer MUST avoid repeated image decoding on frame navigation when a manifest image entry already points to a valid local image file. In that case, the viewer MUST return a resolved file path or an equivalent Gradio-compatible file reference for image outputs.

#### Scenario: 有效图片路径直接返回
- **WHEN** 当前样本的 raw 或 processed image、LiDAR 或 radar 字段解析到存在的本地图片文件
- **THEN** viewer MUST return a path-based image output for the corresponding Gradio image component
- **AND** viewer MUST NOT open and copy the image with PIL for that navigation callback

#### Scenario: 缺失图片保持安全降级
- **WHEN** 当前样本的图片路径缺失、不存在或无法解析
- **THEN** viewer MUST return a safe empty image output
- **AND** the callback MUST NOT fail because of that missing image

### Requirement: 渲染缓存按输出类型拆分
Gradio viewer MUST cache sample render results at a granularity that separates static sample outputs from future distribution controls. Changing future horizon, distribution view, chart type, or show fusion MUST NOT force raw/processed modality image and GPS/mmWave outputs to be recomputed.

#### Scenario: Future 控件变化不重算模态展示
- **WHEN** 用户只修改 Future Beam Distribution Inspector 的 horizon、view type、chart type 或 show fusion
- **THEN** viewer MUST reuse cached raw/processed modality outputs for the current sample when available
- **AND** viewer MUST recompute only the future distribution outputs that depend on the changed controls

#### Scenario: 样本切换命中预热缓存
- **WHEN** 用户切换到已由 preload 或历史访问渲染过的样本
- **THEN** viewer MUST reuse cached render outputs for that sample when controls are unchanged
- **AND** viewer MUST still return the correct sample text and index for the current filtered view

### Requirement: 隐藏 Tab 惰性同步且不改变布局
Gradio viewer MUST keep the existing page layout and Tab structure while reducing frame-navigation output volume. On frame navigation, the viewer MUST update the current visible Tab and global sample state; hidden Tabs MAY be refreshed lazily when the user selects them.

#### Scenario: 切帧不更新隐藏 Tab
- **WHEN** 用户在某个 Tab 中点击上一帧、下一帧、拖动 slider 或触发 timer tick
- **THEN** viewer MUST update the outputs visible in the active Tab for the new sample
- **AND** viewer MUST NOT be required to synchronously return every hidden Tab output in the same callback

#### Scenario: 打开隐藏 Tab 时刷新到当前样本
- **WHEN** 用户切换到之前未同步的 Tab
- **THEN** viewer MUST render that Tab using the current sample index, scene, split, show mode and diagnostic controls
- **AND** the newly visible Tab MUST NOT display a stale sample after the select callback completes

#### Scenario: 页面布局保持不变
- **WHEN** 性能优化启用后 viewer 页面加载完成
- **THEN** the page MUST still contain the existing top controls and the Overview, Raw Modalities, Processed Modalities and Diagnostics Tabs
- **AND** the optimization MUST NOT move, remove, rename or visually restructure those controls and Tabs

### Requirement: 性能诊断可观测
Gradio viewer MUST provide an optional performance profiling mode that reports callback timing and output-volume information without adding visible page components.

#### Scenario: 启用性能日志
- **WHEN** 用户使用性能 profiling mode 启动 viewer
- **THEN** each navigation, filter, future-control or tab-select callback MUST log timing information for filtering, cached render lookup, render computation and total callback time
- **AND** each log entry MUST include the event type and returned component count

#### Scenario: 默认不输出性能噪声
- **WHEN** 用户未启用性能 profiling mode
- **THEN** viewer MUST keep the normal console output behavior
- **AND** performance logging MUST NOT change page layout or callback return values
