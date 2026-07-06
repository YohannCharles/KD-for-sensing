# Copilot Instructions

This repository uses a thin, shared agent context. Do not infer project rules from this file alone.

- Start with `AGENTS.md`, then `docs/agent_navigation.md`.
- Use `docs/agent_context/README.md` for task-scoped context.
- Use `docs/current_research_brief.md` only as a short orientation note; formal claim status lives in `docs/result_claims_registry.md`, and protocol details live in `docs/experiment_protocols.md`.
- Non-trivial feature, architecture, training-flow, data-contract, or compatibility work goes through OpenSpec.
- All project Python commands use `conda run -n kd_mm_beam ...`.
- Do not recommend committing `dataset/`, `outputs/`, `outputs/cache/`, `logs/`, checkpoints, metrics, cache files, or generated reports.
- Keep generated suggestions within the current `src/kd_sensing` package structure and do not add compatibility facades or old entrypoints.
