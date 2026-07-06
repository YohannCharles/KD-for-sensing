# CLI 与 scripts 任务上下文

用于 package console script、包内 CLI、help smoke、`scripts/` 本地 runner、verify helper 和入口生命周期改动。

## 先读

- README 的主要入口和破坏性变更章节
- `openspec/specs/project-entrypoint-lifecycle/spec.md`
- `openspec/specs/project-architecture/spec.md`
- `docs/project_surface_inventory.md` 的入口生命周期和 scripts 分类
- `src/kd_sensing/diagnostics/cli_surface.py` 的 public console script lifecycle 清单

## Owner

- Console scripts：`pyproject.toml`
- Console script lifecycle guardrail：`src/kd_sensing/diagnostics/cli_surface.py`
- 包内 CLI：`src/kd_sensing/cli/`
- 真实 workflow owner：对应 `src/kd_sensing/diagnostics/`、`src/kd_sensing/baselines/`、`src/kd_sensing/data/` 或 `src/kd_sensing/engine/`
- 本地/manual runner：`scripts/`

## 边界

- CLI 文件应保持 thin parser/IO glue；核心逻辑放回 owner module。
- 不新增旧顶层脚本、退役 console script、package facade 或绕过 `src/kd_sensing` 的入口。
- `scripts/` 中的本地 runner 必须在 inventory 中说明 lifecycle、输出边界和删除条件。
- Scene31-34 encoder ablation 使用 `scripts/generate_scenes31_34_encoder_ablation.py --family tinyvit|patchvit` 和唯一 family/manifest runner；不要新增 PatchViT 专用 shell。
- 所有项目 Python 命令使用 `conda run -n kd_mm_beam ...`。

## 验证

- `conda run -n kd_mm_beam pytest tests/test_cli_help.py -q`
- `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`
- `conda run -n kd_mm_beam kd-sensing-project-surface-doctor --scope cli-surface --format markdown --fail-on error`
- `conda run -n kd_mm_beam python scripts/verify_compile.py`
- 聚合入口：`make verify-cli-config`、`make verify-compile`
