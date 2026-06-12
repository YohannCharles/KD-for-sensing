## 1. 支持面和入口盘点

- [x] 1.1 用 `rg`/AST 静态引用确认 `tools/visualization`、旧静态 modality visualization、Top8 selector dataset、GPS window、DeepVerse/DT31 和退役 JEPA 配置的源码/配置/文档引用点；BeamBench 只盘点为保留面，不列入删除候选。
- [x] 1.2 确认保留面清单：`jepa_context_image`、`GPSQueryPool`、paired baseline/control、vision-position baseline suite、完整 BeamBench 相关代码/Arnold22 Camera AE+GPS Direct、viewer manifest CLI 和 `jepa_visual_analysis`。

## 2. 删除退役源码与配置

- [x] 2.1 确认 `tools/visualization/` 删除并移除架构 allowlist、README 和 inventory 中的 viewer support 描述。
- [x] 2.2 删除旧静态 modality visualization workflow、对应测试和旧 PNG 总览图文档引用，同时保留 viewer manifest 导出 alias。
- [x] 2.3 删除 DeepSense6G Top8 selector dataset 和 GPS window baseline 源码/CLI/config；不删除、不改写任何 BeamBench 相关源码、脚本、配置或测试。
- [x] 2.4 删除 DeepVerse/DT31 generator、label builder、split/sanity check、generation config 和当前入口文档引用，不清理本地数据或产物。
- [x] 2.5 删除退役 JEPA 实验配置：scene31-only、非 BeamBench 的 last-checkpoint、next-beam downstream ablation 等非主线 YAML；保留 `beambench_fair` 相关配置。
- [x] 2.6 删除小型孤立模块和空目录引用，例如 `target_shot_runtime.py`、空 model 子目录和无用 dataset rule config。

## 3. 同步注册、文档和规格

- [x] 3.1 更新 `pyproject.toml`、registry/migration guard、CLI help 测试和架构边界测试，使退役入口不再作为当前脚本或 allowlist。
- [x] 3.2 更新 README、`docs/project_surface_inventory.md` 和 `docs/experiment_matrix.md`，明确 Image+GPS JEPA query-pool 主线、保留 baseline/control 和退役面。
- [x] 3.3 更新 `openspec/specs/project-architecture/spec.md`，使当前架构要求与本 change 的 delta 一致。

## 4. 验证

- [x] 4.1 运行 `openspec validate prune-to-jepa-query-pool-surface --strict` 和 `openspec status --change prune-to-jepa-query-pool-surface`。
- [x] 4.2 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
- [x] 4.3 运行 `conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_config_load_characterization.py -q`。
- [x] 4.4 根据实际触碰面运行 focused tests；如果删除旧测试导致无需运行，记录原因。
- [x] 4.5 检查 `git status --short`，确认没有新增本地数据、输出、日志、cache 或 checkpoint 纳入源码变更。
