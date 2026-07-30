# CLI 任务上下文

public CLI 只有 `kd-sensing-train`、`kd-sensing-evaluate`、`kd-sensing-preprocess`。PCPF-T、trajectory 和 CSI/TSPC 使用 `tools/`/`scripts/` 下的本地 helper，不注册 console script，也不成为 canonical MMW route。

不要新增历史 wrapper、console script 或兼容入口。先读 `openspec/specs/repo-boundaries/spec.md`。

最小验证：`conda run -n kd_mm_beam pytest tests/test_cli_help.py tests/test_architecture_boundaries.py -q`。
