## REMOVED Requirements

### Requirement: Gradio 交互入口
**Reason**: 仓库级 Gradio Blocks viewer 退役，`tools/visualization/gradio_multimodal_viewer.py` 不再作为当前入口维护。
**Migration**: 使用 `kd-sensing-export-viewer-manifest` 生成 manifest；需要交互浏览时由外部/历史工具消费 manifest。

### Requirement: Manifest 数据格式与路径解析
**Reason**: 该要求原属于 Gradio viewer runtime；manifest 字段和路径解析保留在 `modality-visual-diagnostics` 的 manifest 导出契约中。
**Migration**: 使用包内 viewer manifest 导出能力和 `viewer_manifest_*` helper。

### Requirement: 样本过滤与时间浏览
**Reason**: Gradio viewer 页面过滤和时间浏览不再维护。
**Migration**: 不迁移到当前仓库入口。

### Requirement: Raw 与 processed 多模态展示
**Reason**: Gradio viewer 展示层退役；仓库仅保留 raw/processed asset manifest 生成。
**Migration**: Manifest 中继续记录 raw/processed asset、label、statistics 和 prediction bundle 字段。

### Requirement: 诊断信息展示
**Reason**: Gradio viewer UI 诊断面板退役。
**Migration**: 使用 manifest JSON、prediction export 文件和 JEPA visual analysis report/table 进行离线诊断。

### Requirement: Future beam 分布诊断
**Reason**: Gradio Diagnostics Tab 退役。
**Migration**: 模型预测导出仍可写出 future beam distribution；JEPA visual analysis 负责论文图和机制诊断。

### Requirement: 页面布局与空状态
**Reason**: Gradio 页面不再属于当前支持面。
**Migration**: 不迁移。

### Requirement: 文档与可选依赖
**Reason**: `tools/visualization/README.md`、sample manifest 和 viewer requirements 随仓库级 viewer support 删除。
**Migration**: README 只保留 manifest 导出 CLI 与 JEPA visual analysis 命令。

### Requirement: Viewer manifest helper 拆分兼容
**Reason**: Gradio viewer 兼容要求退役；manifest helper 拆分仍由 `modality-visual-diagnostics` 和 project architecture 约束。
**Migration**: 使用 `src/kd_sensing/diagnostics/viewer_manifest*.py` 作为包内 manifest 导出实现。
