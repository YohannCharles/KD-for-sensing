## 1. 盘点与保护边界

- [x] 1.1 用 `rg -n "hist_beam|HiST|kd-sensing-hist-beam|configs/hist_beam" README.md pyproject.toml configs scripts src tests docs openspec/specs` 生成 Hist 当前支持面清单。
- [x] 1.2 盘点 `src/kd_sensing/engine`、`src/kd_sensing/models`、`src/kd_sensing/evaluation` 中 Hist 专用文件、导入方、registry 名称和 training extension 挂钩。
- [x] 1.3 盘点 `outputs/` 大目录、Hist/P3/V8/V9/debug/smoke/plan-check/stale 候选和当前主线应保护目录，记录 `gps_window_*hist2` 等不能裸字符串误删的例外。
- [x] 1.4 确认工作区已有用户侧改动，不 revert 非本 change 文件；实施前只列出本 change 需要触碰的文件集。

## 2. Hist 源码与配置退役

- [x] 2.1 删除 `configs/hist_beam/` 及引用该目录的 shell wrapper、README 命令和测试 fixture，不新增 virtual config alias。
- [x] 2.2 删除 `src/kd_sensing/cli/hist_beam_loso.py` 和 `pyproject.toml` 中 `kd-sensing-hist-beam-loso` console script 声明。
- [x] 2.3 删除 `src/kd_sensing/engine/hist_beam_*` 专用模块，并清理 `engine/__init__.py`、training extension、profiling helper、run metadata、summary 或其它导入方。
- [x] 2.4 删除 `src/kd_sensing/models/fusion/hist_beam.py`、Hist variants 构建逻辑和 `hist_beam_fusion` 注册名，并清理 `models/__init__.py`、`models/fusion/__init__.py` 和默认组件导入。
- [x] 2.5 删除 Hist 专用 evaluation helpers、prediction artifact writer、residual writer 和相关 output schema 挂钩。
- [x] 2.6 保留当前主线模型、engine 和 evaluation 模块；确认 GPS candidate、Top8 selector、BGAM、camera residual、MMW GPS v2、CSI、Raymobtime 和 viewer workflow 不依赖被删 Hist 模块。

## 3. 测试与注册错误收口

- [x] 3.1 删除或改写 `tests/test_hist_beam_*`、history-anchor Hist、image-only Hist、V7/V8/V9 Hist、Hist CLI help 和 Hist training IO 断言。
- [x] 3.2 新增或更新 registry/config 测试，验证 `hist_beam_fusion`、`configs/hist_beam/*` 和 Hist variants 被拒绝或不可用，且错误信息说明研究线已退役。
- [x] 3.3 更新架构边界测试，验证当前源码不再引用 `kd_sensing.engine.hist_beam_*` 或 `kd_sensing.models.fusion.hist_beam`。
- [x] 3.4 更新 runtime cleanup 测试，覆盖退役 Hist 输出候选、debug/plan-check 候选、protected checkpoint 和 `gps_window_*hist2` 非裸字符串误删规则。

## 4. 文档、规格和输出结构

- [x] 4.1 更新 README quickstart、主要入口、数据和产物边界，删除 HiST-Beam 当前运行命令，并说明 Hist 研究线已退役。
- [x] 4.2 更新 `docs/experiment_matrix.md`、`docs/project_surface_inventory.md`、`docs/research_notes.md` 和相关 docs，移除 Hist 推荐工作流或标记为历史记录。
- [x] 4.3 更新 `pyproject.toml` description 和 scripts，使项目描述聚焦当前保留主线且不暴露 Hist CLI。
- [x] 4.4 更新当前 OpenSpec specs 或保留本 change delta，确保当前支持契约不再要求 HiST-Beam/Hist 能力。
- [x] 4.5 更新当前主线配置或文档中的输出目录约定，避免新实验默认写入 `outputs/other/` 或 `outputs/` 根目录。

## 5. outputs 清理执行

- [x] 5.1 使用 `conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --root outputs --root logs --manifest outputs/cleanup_manifests/runtime_cleanup_<timestamp>.json` 生成 dry-run manifest。
- [x] 5.2 审计 manifest，确认 Hist/P3/V8/V9/debug/smoke/plan-check/stale 候选原因、大小、风险等级和 protected 状态；确认当前主线 analysis/cache/features/best checkpoints 被保护。
- [x] 5.3 如 manifest 缺少退役 Hist 候选分类，先更新 cleanup 规则和测试，再重新生成 manifest。
- [x] 5.4 使用 `conda run -n kd_mm_beam kd-sensing-clean-runtime-artifacts --delete --manifest <manifest> --confirm-delete` 删除未受保护候选，并保存 deletion report。
- [x] 5.5 删除后重新运行 `conda run -n kd_mm_beam kd-sensing-runs --outputs outputs --logs logs --format json --output outputs/cleanup_manifests/run_index_after_cleanup.json`，确认剩余输出结构可审计。

## 6. 验证

- [x] 6.1 运行 `openspec validate cleanup-outputs-and-remove-hist --strict`。
- [x] 6.2 运行 `openspec status --change cleanup-outputs-and-remove-hist`，确认 apply 所需 artifact 完整。
- [x] 6.3 运行 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py tests/test_component_registry.py tests/test_runtime_artifact_cleanup.py -q`。
- [x] 6.4 运行保留 CLI smoke：`conda run -n kd_mm_beam kd-sensing-train --help`、`conda run -n kd_mm_beam kd-sensing-evaluate --help`、`conda run -n kd_mm_beam kd-sensing-preprocess --help`、`conda run -n kd_mm_beam kd-sensing-export-viewer-manifest --help`、`conda run -n kd_mm_beam kd-sensing-visualize-modalities --help`。
- [x] 6.5 运行当前主线 focused tests：`conda run -n kd_mm_beam pytest tests/test_raymobtime_s008_selection.py tests/test_modality_visual_diagnostics.py tests/test_top8_selector_runner.py -q`，若测试文件在当前工作区不存在则记录原因并选择等价保留 workflow 测试。
- [x] 6.6 最终运行 `conda run -n kd_mm_beam pytest -q`，并记录任何因外部数据、环境或用户侧未提交改动导致无法完成的验证。
