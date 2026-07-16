# CLI 任务上下文

public CLI 只有 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`，训练和评估读取 MMW 或 DeepSense6G canonical config。`scripts/` 只保留服务 MMW all-weather、T2 screening、BPA/CMA 与 summary 的本地 helper，不能变成额外 public CLI。

先读 `openspec/specs/project-entrypoint-lifecycle/spec.md`。不要恢复任何额外 public CLI 或历史 helper。

最小验证：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。
