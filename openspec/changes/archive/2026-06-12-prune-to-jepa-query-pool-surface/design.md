## Context

当前仓库已经完成 JEPA downstream、GPS-query pooling、vision-position baselines 和 JEPA visual analysis 的主要实现，但历史研究线仍以源码、配置、文档、entry point 或架构 allowlist 的形式留在当前表面积里。尤其是 `tools/visualization/` 已在工作树中被删除，而 README、inventory 和测试仍把它描述成当前 viewer 入口；旧静态 modality visualization、GPS window baseline、DeepVerse/DT31 和部分 JEPA ablation 配置也会让后续维护误判主线。

本 change 以源码和文档收口为目标，不迁移、不删除真实数据或本地实验产物。所有验证命令继续使用 `conda run -n kd_mm_beam ...`。

## Goals / Non-Goals

**Goals:**

- 明确当前主线：Image+GPS JEPA query-pool、paired baseline/control、vision-position baseline suite 和 JEPA visual analysis。
- 删除 P0/P1 退役源码、配置、entry point、测试和文档引用，并把 DeepVerse/DT31 作为本轮退役面处理。
- 保留仍支撑主线的 `jepa_context_image`、`GPSQueryPool`、`vision_position`、`jepa_visual_analysis` 和 viewer manifest 导出 CLI。
- 让 README、inventory、OpenSpec 和架构边界测试对同一支持面达成一致。

**Non-Goals:**

- 不删除 `dataset/`、`outputs/`、`logs/`、cache、checkpoint 或 `All_models/` 历史权重。
- 不在本轮退役 MMW GPS v2、BGAM 或 CSI hardening 的全部实现；这些属于更大 P2 收口，需要单独改 README/OpenSpec 和实验结论。
- 不删除或改写任何 BeamBench 相关源码、配置、脚本、测试或入口；Arnold22 Camera AE+GPS Direct 和现有 BeamBench wrapper 保持当前状态。
- 不重写 JEPA 下游模型、训练 loop 或视觉分析数值逻辑。
- 不归档 `add-vision-position-baselines` 和 `add-jepa-visual-analysis-suite`；如需 archive，可在本轮实现后单独执行。

## Decisions

- 当前支持面用文档和架构测试双重约束。README 负责快速上手口径，`docs/project_surface_inventory.md` 负责可审计表面积，`tests/test_architecture_boundaries.py` 拒绝退役入口回流。
- 旧 viewer support 采用确认删除而不是迁移。包内 `kd-sensing-export-viewer-manifest` 继续生成 manifest，`kd-sensing-visualize-modalities` 仅作为薄 alias；仓库级 Gradio viewer 文件不再是当前维护对象。
- 旧静态 modality visualization 采用源码删除而不是保留兼容 facade。该 workflow 与当前 manifest/JEPA analysis 出口重复，且会牵出 matplotlib 渲染和样本 PNG 总览图维护负担。
- GPS window、DeepVerse/DT31 和 Top8 selector dataset 采用整条路线删除。BeamBench 相关代码不进入本轮删除范围，避免影响 Arnold22 Camera AE+GPS Direct 和现有复现辅助入口。
- JEPA 配置缩面只保留论文主线、必要控制组和 `beambench_fair` 相关配置。删除 scene31-only、非 BeamBench 的 last checkpoint、next-beam downstream ablation 等非主线配置，避免实验矩阵继续横向扩张。

## Risks / Trade-offs

- 退役入口可能仍被某些本地脚本或历史笔记引用 → 通过 `rg`、架构边界测试和 CLI/config smoke 捕获源码级引用；历史文档只保留为历史背景时必须标明非当前入口。
- 删除配置会影响复现历史 ablation → README 和 inventory 记录主线保留口径；历史实验产物不删除，必要时可从 git 历史恢复配置。
- BeamBench 仍被当前实验依赖 → 本轮不修改 BeamBench 相关源码、配置、脚本、测试或入口，后续若要收口需单独开 change 并先确认 Arnold22 Camera AE+GPS Direct 需求。
- DeepVerse/DT31 与 MMW 准备模块可能共享命名或文档 → 本轮只删除 `deepverse` 专属 generator/label/split/sanity check/config/test 引用，不触碰 `data/mmw` 当前准备模块。
- 已有工作树很脏 → 实现时不回滚用户已有改动，只在目标文件上做增量编辑，并在最终说明中列出本轮触碰范围。
