## 1. 基线与可观测性

- [x] 1.1 在 `gradio_multimodal_viewer.py` 新增默认关闭的 `--profile-render` 参数，用结构化日志记录 callback event、filter/render/cache/total 耗时和返回组件数量。
- [x] 1.2 为 `render_sample()` 或新渲染路径添加可注入计时器/计数器，便于测试 cache hit、图片解码次数和过滤调用次数。
- [x] 1.3 使用示例 manifest 或小型测试 manifest 记录优化前后基线，命令必须使用 `conda run -n kd_mm_beam python ...`。

## 2. 过滤与导航缓存

- [x] 2.1 实现 `FilteredSampleIndex` 或等价 helper，按 `(scene, split, show_mode)` lazy 缓存过滤后的 sample index 列表。
- [x] 2.2 改造 slider、上一帧、下一帧、timer tick 和 filter change 回调，使非过滤事件复用 cached filtered indices。
- [x] 2.3 增加测试覆盖同一过滤条件下连续导航不重复扫描完整 manifest，过滤条件变化后刷新缓存。

## 3. 图片与渲染缓存

- [x] 3.1 新增路径优先图片输出 helper，存在的 raw/processed 图片返回 Gradio 可接受的文件路径，缺失或坏路径返回安全空值。
- [x] 3.2 将 `_SampleRenderCache` 拆分为样本静态输出缓存和 Future Beam Distribution 输出缓存，避免 future 控件变化重算 raw/processed 模态。
- [x] 3.3 调整 preload 逻辑，使其预热拆分后的缓存，并保持 LRU 上限和后台线程异常吞掉策略。
- [x] 3.4 增加测试覆盖图片路径输出不触发 PIL 解码、future 控件变化复用静态输出、预热样本切换命中缓存。

## 4. Tab 惰性同步

- [x] 4.1 在 Gradio Blocks 中加入 active tab state 和 Tab select 事件，页面布局、Tab 名称和组件位置不得改变。
- [x] 4.2 将当前全量 `canonical_outputs + mirrored_outputs` 回调拆为按 Tab 分组的输出更新：切帧只更新当前可见 Tab 和全局状态，Tab select 时刷新目标 Tab。
- [x] 4.3 确保 Overview、Raw Modalities、Processed Modalities、Diagnostics 四个 Tab 在被选中后显示当前 sample index 对应内容，不出现可见旧样本。
- [x] 4.4 增加 Gradio build/smoke 测试覆盖 Tab state 初始化、切帧输出分组和 Tab select 刷新路径。

## 5. 文档与验证

- [x] 5.1 在 `tools/visualization/README.md` 记录 `--profile-render` 用法、性能边界和“不调整页面布局”的实现约束。
- [x] 5.2 运行 `conda run -n kd_mm_beam pytest tests/test_modality_visual_diagnostics.py` 验证 viewer 相关测试。
- [x] 5.3 如本地 Gradio 依赖可用，运行 `conda run -n kd_mm_beam python tools/visualization/gradio_multimodal_viewer.py --manifest tools/visualization/sample_manifest_example.json --check-only --profile-render` 做 smoke 验证。
- [x] 5.4 汇总优化结果，明确说明哪些延迟已降低、哪些剩余延迟属于 Gradio 前端大量组件渲染的固定成本。
