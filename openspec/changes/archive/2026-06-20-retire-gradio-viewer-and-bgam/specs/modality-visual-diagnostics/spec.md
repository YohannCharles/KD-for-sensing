## MODIFIED Requirements

### Requirement: 诊断入口与配置
系统 MUST 将模态诊断入口收敛到仍保留的 JEPA visual analysis、GPS shortcut benchmark 和其它明确 current 的非 viewer 诊断能力。旧静态 PNG 报告入口、仓库级 Gradio viewer support、`tools/visualization/` 启动路径、viewer manifest 导出 CLI、`kd-sensing-visualize-modalities` 兼容 alias 和 viewer manifest/prediction/cache/schema helper MUST 不再作为可运行工作流、安装入口或 supporting helper 保留。

#### Scenario: 旧 viewer 和 manifest 命令被拒绝
- **WHEN** 用户运行旧的 `the retired modality visualization command`、`the retired script entry`、仓库级 Gradio viewer 启动路径、`kd-sensing-export-viewer-manifest` 或 `kd-sensing-visualize-modalities`
- **THEN** 系统 MUST 拒绝该入口或不再提供该入口
- **AND** 错误信息或文档 MUST 指向仍保留的 `kd-sensing-jepa-visual-analysis`、`kd-sensing-jepa-gps-shortcut-benchmark` 或其它 current 诊断入口

#### Scenario: 诊断配置不恢复 viewer manifest
- **WHEN** 用户加载包含旧 `diagnostics.visualization` 或 viewer manifest 字段的配置
- **THEN** 配置解析 MUST 不启动 viewer manifest 导出或 Gradio viewer
- **AND** 若该字段只服务退役 viewer workflow，系统 MUST 早失败或忽略并记录清晰退役提示

## REMOVED Requirements

### Requirement: 复用真实处理后张量
**Reason**: 该 requirement 描述 viewer manifest 数据准备和 viewer prediction export；viewer manifest 已退役。
**Migration**: JEPA visual analysis 和其它保留诊断由各自 specs 约束真实输入来源。

### Requirement: 只读诊断行为
**Reason**: 该 requirement 的 manifest 导出、viewer 输出目录和 image motion cache 边界随 viewer manifest 退役。
**Migration**: 保留诊断仍必须遵守本地产物边界和各自只读要求。

### Requirement: 可视化诊断内部轻量模块边界
**Reason**: 该 requirement 约束 `viewer_manifest_*` helper；这些 helper 不再作为 current 诊断边界维护。
**Migration**: 若少量通用 JSON/asset helper 被保留，必须迁入非 viewer 命名模块并由对应 current spec 约束。

### Requirement: Manifest 行为保持兼容
**Reason**: viewer manifest payload、prediction bundle 和 `viewer_command` 兼容性不再维护。
**Migration**: 无兼容迁移；外部查看器不再是仓库内 current 支持面。
