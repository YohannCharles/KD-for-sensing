# Claude Project Context

本文件是 Claude Code 的薄适配入口。完整项目规则仍以 `AGENTS.md`、`docs/agent_navigation.md`、`docs/agent_context/README.md` 和当前 OpenSpec specs 为准。

## Load First

- Read `AGENTS.md` for operating rules, `kd_mm_beam`, OpenSpec, and local artifact boundaries.
- Read `docs/agent_navigation.md` before non-trivial changes.
- Use `docs/agent_context/README.md` to select only the scoped context needed for the task.
- Read README and the three current OpenSpec specs for the current surface.

## Claude-Specific Notes

- Keep this file short. Do not copy the full route table, full OpenSpec requirements, full retired-route list, or full claim table here.
- All project Python commands, including tests, use `conda run -n kd_mm_beam ...`.
- Do not treat `dataset/`, `outputs/`, `outputs/cache/`, `logs/`, checkpoint files, or local metrics as source changes.
