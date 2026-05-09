## Context

`replace-visual-diagnostics-with-gradio` 已经把静态诊断迁移为 Gradio viewer，并在 `tools/visualization/gradio_multimodal_viewer.py` 中实现了 render cache 和邻近样本 preload。但当前切帧路径仍然偏重：每个 slider/prev/next/timer 回调都会重新执行 `filter_samples()`，渲染 raw/processed 图片、GPS/mmWave Plotly 图、confidence/quality/gate 图表、future distribution 图表和多个 DataFrame/JSON，然后同时返回 Overview 与其他 Tab 的镜像输出。

用户观测到下一帧约 `0.5s + 1.1s`、跨 20 帧约 `2.0s + 1.7s`。这类延迟通常由两部分组成：后端 Python 回调计算/序列化，以及浏览器端接收大量 image/Plotly/DataFrame 更新后的渲染。后端重复过滤、图片解码和图表重建可以优化；Gradio 前端大量组件更新的固定成本只能通过减少同步输出范围来降低，无法完全消除。

约束是不能调整页面布局：顶部控件、Overview/Raw Modalities/Processed Modalities/Diagnostics Tab、组件位置和展示内容都保持不变。

## Goals / Non-Goals

**Goals:**

- 降低切换下一帧、上一帧、slider 跳转和自动播放 tick 的后端回调耗时。
- 降低每次切帧同步返回给 Gradio 前端的输出数量和序列化体积。
- 保持页面布局、Tab 结构、控件位置和 manifest 格式兼容。
- 保持缺失模态、坏路径、坏 JSON 和空过滤结果的安全降级行为。
- 提供可选性能观测能力，用于确认瓶颈来自过滤、渲染、序列化还是前端更新。

**Non-Goals:**

- 不替换 Gradio、不引入 React/FastAPI 自定义前端。
- 不改变 raw/processed/diagnostics 的展示语义。
- 不在页面内运行模型推理或训练。
- 不承诺所有机器、所有浏览器和所有 manifest 规模下达到固定毫秒级延迟。

## Decisions

### Decision 1: 为过滤结果建立 lazy index cache

新增一个小型 `FilteredSampleIndex` 或等价结构，在 `build_demo()` 时持有样本列表，并按 `(scene, split, show_mode)` 缓存过滤后的样本 index 列表。过滤控件变更时计算一次，slider/prev/next/timer 只复用缓存结果。

理由：
- 当前 `filter_samples(samples, ...)` 每次回调都线性扫描完整 manifest。跨 20 帧跳转时如果 manifest 较大，这部分会被重复放大。
- Manifest 在 viewer 启动后是只读的，不需要复杂失效逻辑。

备选方案是启动时预计算所有 scene/split/show mode 组合。该方案读取更快，但 scene 数量较多时会复制大量 index 列表，内存占用不必要；lazy cache 更保守。

### Decision 2: 图片输出优先返回文件路径

对于 `raw.image`、`raw.lidar`、`raw.radar`、`processed.image`、`processed.lidar`、`processed.radar`，先解析路径并确认文件存在，然后把路径字符串返回给 `gr.Image`；只有确实需要转换时才使用 PIL。

理由：
- 当前每次切帧都会 `Image.open(...).convert("RGB").copy()`，即使浏览器最终只需要展示同一张文件。
- Gradio image 输出可以接收文件路径，减少 Python 侧图片解码和对象复制成本。
- 页面布局和 manifest 格式不变。

备选方案是把图片预加载为 PIL 对象。该方案能减少磁盘 IO，但会占用较多内存，并且仍需序列化图片对象；路径输出更适合已有 viewer asset 文件。

### Decision 3: 拆分渲染缓存粒度

把当前 `_SampleRenderCache` 从“整组 base outputs 按 sample+controls 缓存”拆为更细粒度：

- 样本静态部分：图片路径、GPS/mmWave 图、info JSON、confidence/quality/gate 图和表格。
- Future distribution 部分：只受 horizon、distribution view、chart type、show fusion 影响。
- 空状态输出：统一缓存可复用的空图、空表结构。

理由：
- 切换 future distribution 控件不应导致 raw/processed 模态重算。
- 切帧时如果样本已预热，应直接命中静态渲染缓存。
- 缓存粒度清晰后，测试可以直接验证同一样本重复渲染不会重复加载图片。

备选方案是只扩大现有 cache 容量。该方案无法解决 controls 变化导致整组缓存失效的问题。

### Decision 4: 隐藏 Tab 惰性同步，布局不变

保持 Overview、Raw Modalities、Processed Modalities、Diagnostics 四个 Tab 和内部组件不变，但回调输出按当前选中 Tab 分组：

- 切帧时更新 `sample_text`、slider 状态和当前可见 Tab 的输出。
- 用户切换到其他 Tab 时，根据当前 sample index 和 filter 状态刷新该 Tab。
- 如果某个 Tab 从未打开过，允许它在打开瞬间刷新；用户可见区域不得显示旧样本。

理由：
- 现有代码每次切帧返回 `canonical_outputs + mirrored_outputs`，其中很多组件处于隐藏 Tab，仍然会触发后端序列化和前端处理。
- 惰性同步不改变布局，只改变隐藏组件的刷新时机。
- 对快速浏览下一帧和自动播放，这是最直接降低前端负载的办法。

备选方案是继续同步所有 Tab，但只靠缓存减少后端耗时。该方案无法解决浏览器端大量 Plotly/Image 更新带来的 `+1.1s` 固定成本。

### Decision 5: 增加可选 profile 输出

新增默认关闭的性能诊断开关，例如 `--profile-render`，在回调中打印结构化日志：

- event type：slider、prev、next、timer、filter、tab_select。
- filtered_count、target_index、cache_hit/miss。
- filter_ms、render_static_ms、render_distribution_ms、callback_total_ms。
- returned_component_count。

理由：
- 用户提供的 `0.5s + 1.1s` 已经说明需要区分后端和前端。没有 instrumentation 时只能猜。
- 日志输出不改变页面布局，默认关闭，不影响普通使用。

## Risks / Trade-offs

- 惰性 Tab 同步导致隐藏组件短时间内不是最新样本 → Tab 选中事件立即刷新，用户可见内容必须始终对应当前样本。
- 文件路径输出在不同 Gradio 版本的缓存行为可能不同 → 保留 PIL fallback，并用现有示例 manifest 做 smoke test。
- 过滤 index cache 占用额外内存 → 使用 lazy cache 和 index 列表，避免复制 sample dict。
- 前端仍可能有明显延迟 → 明确这是 Gradio 大量组件更新的下限；通过 profile 输出判断是否已经从后端转移到前端。
- 并发 preload 可能与用户快速点击竞争 → 保持小线程池和 LRU 上限，不在回调中等待后台预热。

## Migration Plan

1. 添加过滤 index cache 和单元测试，确认 filter 控件不变时重复导航不重复扫描完整样本列表。
2. 将图片输出改为路径优先，并保留缺失文件与坏图片 fallback。
3. 拆分 render cache，覆盖 sample 静态输出和 future distribution 输出。
4. 改造 Gradio event binding，加入 active tab state 和 Tab select 刷新逻辑，保持页面布局不变。
5. 添加可选 profile 输出，并在 README 中简要说明如何定位后端和前端耗时。
6. 运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py`；如环境支持 Gradio，再用示例 manifest 做 `--check-only` 或手工启动 smoke 验证。

Rollback 策略：若惰性 Tab 同步在目标 Gradio 版本存在兼容问题，先保留过滤 cache、路径图片输出和拆分 render cache，把 Tab 输出暂时切回全量同步。

## Open Questions

- Gradio 当前安装版本的 Tab select 事件能否稳定区分 active tab id；实现时需要用本地环境确认。
- 用户机器上的 `0.5s + 1.1s` 中前端部分是否主要来自 Plotly，还是图片传输；需要 profile 与浏览器观察共同确认。
