## Why

Gradio viewer 和 BGAM 相关路径已经确认不会再被使用，但当前 README、OpenSpec、console scripts、配置、测试和维护索引仍把 viewer manifest 与 DeepSense6G/MMW BGAM 写成当前支持面。继续保留这些入口会扩大维护面、拖慢架构治理，并让后续主线实验误以为必须兼容一批已无价值的 workflow。

## What Changes

- **BREAKING** 退役所有 BGAM 相关当前能力，包括 DeepSense6G GPS+LiDAR BGAM、MMW Town GPS+LiDAR BGAM、GPS pseudo-history BGAM、BGAM manifest enrich、BGAM dataset/model/loss/engine/CLI/config/tests、BGAM debug mask/report 和文档命令。
- **BREAKING** 退役仓库内 viewer 工作流相关支持面，包括 Gradio viewer 遗留边界、viewer manifest 导出 CLI、`kd-sensing-visualize-modalities` 兼容 alias、viewer manifest/prediction/cache/schema helper、README viewer 运行说明和对应 CLI help/architecture allowlist。
- 更新 current mainline、quickstart、健康检查、项目 surface inventory、maintainer context index 和架构边界，使推荐入口聚焦仍保留的 supervised/adaptation、Image+GPS JEPA、JEPA-MSAC、Vision-Position/Arnold22、MMW GPS v2、CSI hardening、JEPA visual analysis、预处理、run index、cleanup 和通用训练评估能力。
- 删除或改写要求 BGAM/viewer manifest 必须存在的 OpenSpec requirements；历史 archive 可保留为只读演进记录，但不得继续作为当前入口、supporting helper 或验证命令来源。
- 增加退役防回流约束：BGAM 与 viewer manifest 的 console scripts、源码模块、配置文件和 focused tests 不得作为兼容包装、薄 alias 或虚拟配置重新出现。

## Capabilities

### New Capabilities
- 无

### Modified Capabilities
- `project-architecture`: 当前主线、console script allowlist、诊断入口、热点治理和 retired-route guard 不再要求 BGAM 或 viewer manifest 入口存在。
- `experiment-workflow`: quickstart、健康检查、配置驱动 workflow、README 要求和验收命令移除 BGAM 与 viewer manifest。
- `modality-visual-diagnostics`: 诊断能力从 viewer manifest/Gradio 相关导出收口为保留的 JEPA visual analysis 与非 viewer 诊断边界，并退役 viewer manifest 导出契约。
- `deepsense6g-gps-lidar-bgam-reranker`: 整个 DeepSense6G BGAM reranker 能力从 current 退役，不再要求实现、配置、CLI、测试或文档。
- `mmw-town-gps-lidar-bgam-reranker`: 整个 MMW Town BGAM reranker 能力从 current 退役，不再要求实现、配置、CLI、测试或文档。
- `gps-pseudo-label-bgam`: GPS pseudo-history BGAM 输入、评估和防泄漏契约整体退役。
- `modality-aware-data-loading`: 移除 BGAM 专用按需模态加载、manifest column mapping 和 BGAM 防泄漏 dataset 契约。
- `mainline-experiment-documentation`: 主线文档、实验矩阵和结果账本不再要求列出 BGAM 或 viewer manifest。
- `spec-lifecycle-boundaries`: lifecycle 分类与退役边界增加 BGAM/viewer manifest 防回流约束。
- `deepsense6g-gps-top8-candidate-selector`: 若其剩余 supporting 语义仅服务 BGAM，则同步退役相关 candidate manifest 支撑要求。
- `mmw-town-gps-top8-candidate-selector`: 若其剩余 supporting 语义仅服务 BGAM，则同步退役相关 candidate manifest 支撑要求。

## Impact

- 受影响源码：`src/kd_sensing/cli/*bgam*`、`src/kd_sensing/cli/export_viewer_manifest.py`、BGAM data/engine/model/loss 模块、`src/kd_sensing/diagnostics/viewer_manifest*` 与 `viewer_predictions.py`。
- 受影响配置与入口：`configs/*bgam*.yaml`、`pyproject.toml` 中 BGAM 与 viewer manifest console scripts、README 安装/quickstart/Viewer Manifest/BGAM 小节。
- 受影响测试：BGAM focused tests、viewer manifest/CLI help 测试、architecture allowlist、config load characterization 和 modality visual diagnostics tests。
- 受影响治理文档：`docs/maintainer_context_index.yaml`、`docs/project_surface_inventory.md`、主线模型目录、实验矩阵、协议表和相关 OpenSpec specs。
- 不影响：archive 历史记录、本地 `outputs/`/`logs/`/`dataset/` 运行产物、已跟踪复现权重、JEPA visual analysis、MMW GPS v2、CSI hardening、通用训练/评估/预处理和 runtime cleanup 工作流。
