# modality-visual-diagnostics Specification

## Purpose
定义当前模态诊断入口在仓库级 Gradio viewer、viewer manifest 导出和旧 visualization alias 退役后的支持边界，确保 JEPA visual analysis、GPS shortcut benchmark 与其它非 viewer 诊断保持只读、可验证、无兼容后门。

## Requirements
### Requirement: 当前诊断入口不再包含 viewer manifest 或 Gradio viewer
系统 MUST 将当前模态诊断入口限定为 JEPA visual analysis、GPS shortcut benchmark 和其它明确登记的非 viewer 诊断。仓库级 Gradio viewer、viewer manifest 导出、viewer prediction export、`kd-sensing-visualize-modalities` alias、`tools/visualization/` wrapper 和 `diagnostics.visualization` virtual config MUST 已退役并从当前支持面删除。

#### Scenario: 旧 viewer/visualization 入口不存在
- **WHEN** 开发者检查 console scripts、包内 CLI、tools wrapper 和配置目录
- **THEN** 项目 MUST 不声明 `kd-sensing-export-viewer-manifest`
- **AND** 项目 MUST 不声明 `kd-sensing-visualize-modalities`
- **AND** 项目 MUST 不保留 `src/kd_sensing/cli/export_viewer_manifest.py`
- **AND** 项目 MUST 不保留已退役的 `configs/diagnostics/modality_visualization.yaml`

#### Scenario: 旧 viewer 配置快速失败
- **WHEN** 用户传入已退役的 `configs/diagnostics/modality_visualization.yaml` 或包含 `diagnostics.visualization` 的配置
- **THEN** migration guard MUST 早失败
- **AND** 错误信息 MUST 说明 viewer manifest 和仓库级 Gradio viewer 已退役

#### Scenario: diagnostics 包不暴露 viewer helper
- **WHEN** 开发者导入 `kd_sensing.diagnostics`
- **THEN** 该包 MUST 不导出 `export_viewer_manifest` 或 viewer prediction export facade
- **AND** 旧 `kd_sensing.diagnostics.viewer_manifest*` module path MUST 不可导入

### Requirement: JEPA visual analysis 保持当前只读诊断
JEPA visual analysis MUST 继续作为当前模态诊断入口，用于从已存在的模型、配置、manifest 或 mock/smoke 输入生成可审计分析产物。该入口 MUST 保持只读输入行为，不修改训练 checkpoint、训练日志、评估报告或 split CSV，并 MUST 将新产物写入 ignored runtime output 分区。

#### Scenario: JEPA visual analysis help 可用
- **WHEN** 用户运行 `kd-sensing-jepa-visual-analysis --help`
- **THEN** 命令 MUST 可解析并显示当前 JEPA visual analysis 参数
- **AND** 该入口 MUST 不导入旧 viewer manifest 或 Gradio runtime helper

#### Scenario: JEPA visual analysis 只读写出
- **WHEN** 用户运行 JEPA visual analysis
- **THEN** 输入 checkpoint、config、manifest、训练日志和评估报告 MUST 不被修改
- **AND** 输出 MUST 写入 `outputs/visual_analysis/` 或配置指定的 ignored analysis 目录

### Requirement: GPS shortcut benchmark 保持当前诊断
GPS shortcut benchmark MUST 继续作为当前诊断入口，用于比较 JEPA 表征、GPS shortcut、Scenario D image observability 和 Predictive Robustness smoke/canonical manifest。该入口 MUST 由当前 diagnostics owner module 维护，不得依赖 viewer manifest helper、Gradio support 或旧 visualization alias。

#### Scenario: GPS shortcut benchmark help 可用
- **WHEN** 用户运行 `kd-sensing-jepa-gps-shortcut-benchmark --help`
- **THEN** 命令 MUST 可解析并显示当前 benchmark 参数
- **AND** 该入口 MUST 不引用旧 viewer manifest CLI 或 visualization alias

#### Scenario: benchmark 产物归入 analysis 分区
- **WHEN** 用户运行 GPS shortcut benchmark
- **THEN** 输出 MUST 写入 `outputs/analysis/` 或配置指定的 ignored analysis 目录
- **AND** benchmark metadata MUST 记录输入来源、mock/smoke/canonical 状态和验证口径

### Requirement: 诊断入口不再包含 viewer manifest
诊断入口 MUST 收敛为 JEPA visual analysis、GPS shortcut benchmark 和其它明确登记的非 viewer 诊断。仓库级 Gradio viewer、viewer manifest 导出、viewer prediction export、`kd-sensing-visualize-modalities` alias、`tools/visualization/` wrapper 和 `diagnostics.visualization` virtual config MUST 已退役并从当前支持面删除。

#### Scenario: viewer manifest 入口退役
- **WHEN** 开发者检查 pyproject、包内 CLI、tools wrapper 和 diagnostics exports
- **THEN** 项目 MUST 不声明 `kd-sensing-export-viewer-manifest` 或 `kd-sensing-visualize-modalities`
- **AND** 项目 MUST 不保留 `kd_sensing.cli.export_viewer_manifest`
- **AND** 项目 MUST 不保留仓库级 `tools/visualization` viewer support helper

#### Scenario: 旧 visualization 配置被拒绝
- **WHEN** 用户传入已退役的 `configs/diagnostics/modality_visualization.yaml` 或 `diagnostics.visualization`
- **THEN** 配置加载 MUST 早失败
- **AND** 错误信息 MUST 说明 viewer manifest 和仓库级 Gradio viewer 已退役

#### Scenario: JEPA visual analysis 作为论文图出口
- **WHEN** 开发者执行 `conda run -n kd_mm_beam kd-sensing-jepa-visual-analysis --help`
- **THEN** 命令 MUST 正常退出
- **AND** 该入口 MUST 使用 `kd_sensing.diagnostics.jepa_visual_analysis` 生成本地分析 manifest、图表、表格和 report

### Requirement: Viewer manifest helper import surface 已退役
Viewer manifest 内部 helper、配置解析、采样选择、processed asset 写出和模型预测导出模块已退役。项目 MUST 不再要求这些 helper 可导入，并 MUST 通过架构边界检查防止其以轻量 facade 名义回流。

#### Scenario: viewer manifest helper 不可导入
- **WHEN** 开发者执行 `import kd_sensing.diagnostics.viewer_manifest_config`
- **THEN** 导入 MUST 失败
- **AND** 项目 MUST 不保留同名兼容 stub

#### Scenario: viewer sampling helper 不回流
- **WHEN** 开发者导入 `kd_sensing.diagnostics.viewer_manifest_sampling`
- **THEN** 导入 MUST 失败
- **AND** 当前 JEPA visual analysis 或 GPS shortcut benchmark MUST 使用自身 owner module 的采样/写出 helper
