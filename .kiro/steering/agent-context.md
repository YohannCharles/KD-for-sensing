# Agent Context Steering

This Kiro steering file only points to the shared project context.

## Required Sources

- `AGENTS.md`: operating rules, command environment, OpenSpec boundary, local artifacts.
- `docs/agent_navigation.md`: before non-trivial code or documentation changes.
- `docs/agent_context/README.md`: task-scoped context selection.
- `docs/current_research_brief.md`: short research orientation, not a formal claim registry.
- `docs/readonly_agent_roles.md`: optional read-only analysis roles.

## Boundaries

- Use `conda run -n kd_mm_beam ...` for project Python commands.
- Keep training outputs, real data, logs, cache, checkpoints, metrics, figures, and reports under ignored local roots such as `dataset/`, `outputs/`, `outputs/cache/`, and `logs/`.
- Do not let hooks or agents rewrite README, OpenSpec artifacts, `AGENTS.md`, or formal claim documents automatically.
- Do not maintain a full task route, retired-route list, OpenSpec requirement copy, or source-tree mirror in steering files.
