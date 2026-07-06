# Agent Project Knowledge Template

本文件用于 Replit、Lovable、Bolt 或其它只提供 Project Knowledge 文本框的工具。它是可粘贴的短模板，不是第二套项目规则。

## Paste This

This project is `KD-for-sensing`. Before non-trivial changes, read `AGENTS.md`, `docs/agent_navigation.md`, and the task-scoped entry in `docs/agent_context/README.md`. Current research orientation is summarized in `docs/current_research_brief.md`, but formal protocol and claim status live in `docs/experiment_protocols.md` and `docs/result_claims_registry.md`.

All project Python commands use `conda run -n kd_mm_beam ...`. Do not commit or rewrite local data, outputs, logs, cache, checkpoints, generated metrics, generated figures, or generated reports from `dataset/`, `outputs/`, `outputs/cache/`, or `logs/`.

Non-trivial feature, architecture, training-flow, data-contract, compatibility, and public-entry changes require OpenSpec context. Keep suggestions within `src/kd_sensing`; do not create old entrypoints, compatibility facades, full source-tree mirrors, full OpenSpec requirement copies, full retired-route catalogs, or full claim tables.

## Maintenance

If this template feels stale, update the authoritative docs first, then adjust this short pointer. Do not let Project Knowledge tooling auto-rewrite README, OpenSpec, `AGENTS.md`, or formal claim documents.
