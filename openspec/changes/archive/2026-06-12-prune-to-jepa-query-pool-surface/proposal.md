## Why

当前仓库同时保留了 Image+GPS JEPA query-pool 主线、多个历史 baseline、旧 viewer 支持、旧静态可视化、GPS window、DeepVerse/DT31、MMW/BGAM/CSI 等研究面，README、inventory 和架构测试对已删除/待删除入口的描述也出现漂移。为了让后续实验和 AI 协作集中在可投稿的 Image+GPS JEPA query-pool 证据链上，需要先把当前支持面、退役面和源码边界收紧。

## What Changes

- **BREAKING** 将当前推荐支持面收敛为 Image+GPS JEPA query-pool 主线：`jepa_context_image + GPSQueryPool`、paired baseline/control、vision-position baseline suite 和 `jepa_visual_analysis` 论文图/诊断出口。
- **BREAKING** 确认删除仓库级 `tools/visualization/` Gradio viewer 支持文件，并同步 README、inventory 和架构 allowlist，避免后续把 viewer support 补回。
- **BREAKING** 退役旧静态 modality visualization PNG workflow，删除 `src/kd_sensing/diagnostics/visualization/` 和对应测试；保留包内 viewer manifest 导出 CLI 作为兼容薄入口。
- **BREAKING** 退役 DeepSense6G Top8 selector dataset、GPS window baseline、DeepVerse/DT31 数据生成路线和小型孤立模块。
- 缩减 JEPA 实验配置矩阵，保留 query-pool、GPS-biased baseline、supervised/random-best 控制组和 `beambench_fair` 相关配置，删除 scene31-only、非 BeamBench 的 last-checkpoint 和 next-beam ablation 配置。
- 同步 README、`docs/project_surface_inventory.md`、`docs/experiment_matrix.md`、OpenSpec specs、pyproject entry points、registry/guardrail、架构测试和 CLI help 测试。
- BeamBench 相关源码、配置、脚本、测试和入口不纳入本轮删除或改写范围；Arnold22 Camera AE+GPS Direct 及现有 BeamBench wrapper 保持当前状态。
- 不清理 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或历史本地产物；本 change 只处理源码、配置、文档和 OpenSpec artifact。

## Capabilities

### New Capabilities

- 无。

### Modified Capabilities

- `project-architecture`: 当前源码入口、诊断入口、配置生命周期和支持面要求改为 Image+GPS JEPA query-pool 优先，并明确旧 viewer support、静态 PNG 可视化、GPS window、DeepVerse/DT31 等退役路径不再作为当前架构能力。
- `modality-visual-diagnostics`: 诊断能力收敛为 viewer manifest 数据准备和 JEPA visual analysis，移除旧静态 PNG workflow 和仓库级 Gradio viewer 启动要求。
- `gradio-visual-analysis`: 仓库级 Gradio Blocks viewer 退役；manifest JSON/asset 仍由包内 CLI 生成，但不再维护 `tools/visualization` Web UI。
- `gradio-viewer-performance`: Gradio viewer 性能优化能力退役，因为 Web UI 入口不再属于当前支持面。
- `gps-window-baseline-beam-prediction`: GPS window 非神经几何 baseline 退役，不再提供 CLI、配置或评估产物契约。
- `deepverse-dt31-data-generation`: DeepVerse/DT31 数据生成、label builder、split 和 sanity workflow 退役。
- `jepa-next-beam-query-transformer`: JEPA next-beam downstream ablation 矩阵退役，当前 JEPA 下游主线转为 GPS-query pooling 与必要 paired controls。

## Impact

- 影响代码范围：`src/kd_sensing/diagnostics/visualization/`、DeepSense6G Top8 selector dataset、GPS window baseline、DeepVerse/DT31 generator/label builder 相关模块、小型孤立 data/model/config 模块，以及对应 registry、CLI、entry point 和测试 allowlist；不改 BeamBench 相关代码。
- 影响配置范围：`configs/baselines/gps_window_*.yaml`、`configs/deepverse/dt31_generation.yaml`、退役 JEPA 实验特化 YAML 和相关诊断/预处理说明。
- 影响文档范围：README、实验矩阵、项目表面积 inventory、OpenSpec project-architecture delta。
- 影响验证：需要运行 OpenSpec strict 校验、架构边界测试、CLI help/config smoke，并根据实际删除面运行相关 focused tests；如果全量回归因环境或时间不可行，需要在最终说明中记录未覆盖风险。
