# CLI 任务上下文

public CLI 只有 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`。`scripts/` 只保留 MMW all-weather、BPA/CMA、必要 summary 与 compile verification；CMSBL 复用现有 train/evaluate，不增加 wrapper 或 console script。

先读 `openspec/specs/project-entrypoint-lifecycle/spec.md`。不要恢复任何额外 public CLI 或历史 helper。

最小验证：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。
