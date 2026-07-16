# 文档任务上下文

当前文档只描述 MMW T2、S1、AMBER-Full、RMBP-MM。历史路线只写入 `docs/retired_routes.md`，不得在 current 文档中恢复成入口、配置或维护目标。

先读 `AGENTS.md`、导航、维护索引和以下 current specs：`ai-maintainer-navigation`、`maintainer-context-index`、`openspec-document-health`。README 负责上手，OpenSpec 负责契约，inventory 负责当前路径，历史说明负责追溯。

验证：`openspec validate --all --strict` 与 `conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q`。
