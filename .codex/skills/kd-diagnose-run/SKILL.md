---
name: kd-diagnose-run
description: Diagnose local runs, runtime artifacts, JEPA evidence, GPS shortcut behavior, paper exports, or project surface issues without mutating protected artifacts.
license: MIT
---

# kd-diagnose-run

Use this skill for run diagnosis, run index summaries, JEPA visual analysis, GPS shortcut benchmark, project surface doctor, dataset audit, paper export checks, or cleanup manifest planning.

## Required context

1. Read `AGENTS.md`, `docs/agent_navigation.md`, `docs/maintainer_context_index.yaml`, and `docs/agent_context/diagnostics.md`.
2. Read the OpenSpec spec for the diagnostic being used, for example `openspec/specs/jepa-visual-analysis-suite/spec.md`, `openspec/specs/jepa-gps-shortcut-benchmark/spec.md`, or `openspec/specs/experiment-run-index/spec.md`.
3. If diagnosis requires a new public workflow, config contract, metric schema, or artifact lifecycle, use an OpenSpec change before implementation.

## Workflow

1. Identify whether the task is read-only diagnosis, fresh local analysis, benchmark execution, paper export, dataset audit, or cleanup planning.
2. Prefer existing package CLIs such as `kd-sensing-runs`, `kd-sensing-jepa-visual-analysis`, `kd-sensing-jepa-gps-shortcut-benchmark`, `kd-sensing-project-surface-doctor`, and `kd-sensing-paper-export`.
3. Keep cleanup as a two-stage manifest workflow; deletion requires explicit user confirmation and the project cleanup command's confirmation flags.
4. Do not turn diagnostic output into reviewed claims without the claim update workflow.

## Commands and artifacts

- Run project Python commands through `conda run -n kd_mm_beam ...`.
- Diagnostics may read local `outputs/`, `logs/`, checkpoints, or manifest paths when the user asks, but generated reports, figures, cache, CSV, JSON, and ledgers stay in ignored `outputs/` or `logs/`.
- Never commit real `dataset/` contents or new checkpoint files.
- Focused validation examples:

```bash
conda run -n kd_mm_beam pytest tests/test_jepa_visual_analysis.py -q
conda run -n kd_mm_beam pytest tests/test_runtime_artifact_cleanup.py tests/test_run_index.py -q
conda run -n kd_mm_beam pytest tests/test_project_surface_doctor.py -q
```
