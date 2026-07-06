# 诊断任务上下文

用于 run index、JEPA visual analysis、GPS shortcut benchmark、project surface doctor、paper export、dataset audit、cleanup manifest 和本地分析脚本。

## 先读

- `openspec/specs/experiment-run-index/spec.md`
- `openspec/specs/jepa-visual-analysis-suite/spec.md`
- `openspec/specs/jepa-gps-shortcut-benchmark/spec.md`
- `openspec/specs/project-health-guardrails/spec.md`
- README 中诊断入口、论文表格、dataset audit 和清理 manifest 章节
- `docs/project_surface_inventory.md` 的诊断热点和 scripts 分类

## Owner

- Diagnostics package：`src/kd_sensing/diagnostics/`
- CLI glue：`src/kd_sensing/cli/`
- 本地分析脚本：`scripts/analysis/`、`scripts/mmw/`
- Paper/claim artifacts：`docs/result_claims_registry.md`、`outputs/paper_artifacts/`

## 边界

- 大多数诊断默认只读本地 `outputs/`、`logs/`、manifest 或用户提供 checkpoint。
- 新生成图表、CSV、HTML、report、cache、ledger 和 checkpoint sidecar 默认写 ignored runtime root，不提交。
- `kd-sensing-research-dashboard --output-html` 生成静态 HTML evidence dashboard；它只展示 candidate-only/readiness 证据和 next action，不启动服务、不升级 claim，也不写正式 claim registry。
- `kd-sensing-research-preview` 生成无训练 preview manifest、dashboard HTML、静态 evidence QA 和 budget manifest；默认不读取真实 `dataset/`、不加载 checkpoint、不写训练产物，真实长跑预算实例写 ignored `outputs/analysis/` 或用户显式路径。
- `kd-sensing-clean-runtime-artifacts` 默认只生成 dry-run manifest；真正删除必须显式确认。
- 诊断证据可以支撑 claim gate，但不能自动把 draft candidate 写成 reviewed claim。

## 验证

- `conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q`
- `conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_run_index.py -q`
- `conda run -n kd_mm_beam pytest tests/test_project_surface_doctor.py -q`
- `conda run -n kd_mm_beam pytest tests/test_research_run_preview.py -q`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
