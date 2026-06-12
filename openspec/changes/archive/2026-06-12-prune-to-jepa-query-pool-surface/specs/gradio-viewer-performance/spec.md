## REMOVED Requirements

### Requirement: 切帧路径缓存过滤结果
**Reason**: 仓库级 Gradio viewer Web UI 不再属于当前支持面，因此不再维护 viewer session 导航性能契约。
**Migration**: 使用 `kd-sensing-export-viewer-manifest` 生成 manifest；论文图和诊断使用 `kd-sensing-jepa-visual-analysis`。

### Requirement: 图片输出避免重复解码
**Reason**: Gradio viewer runtime 已退役，图片回调性能优化不再适用。
**Migration**: Manifest 导出仍可写出 raw/processed asset 路径供外部 viewer 或离线分析读取。

### Requirement: 渲染缓存按输出类型拆分
**Reason**: Gradio viewer runtime 已退役，Tab/控件级渲染缓存不再维护。
**Migration**: 不迁移；离线 manifest 与 JEPA analysis 产物默认写入 ignored 输出目录。

### Requirement: 隐藏 Tab 惰性同步且不改变布局
**Reason**: Gradio viewer 页面布局和 Tab 同步不再属于仓库当前能力。
**Migration**: 不迁移。

### Requirement: 性能诊断可观测
**Reason**: Gradio callback profiling 随 Web UI 退役。
**Migration**: 训练/评估性能仍由 training throughput 和 run metadata 能力覆盖。
