---
name: kd-archive-change
description: Close out and archive completed KD-for-sensing OpenSpec changes while preserving validation, source, and artifact boundaries.
license: MIT
---

# kd-archive-change

Use this skill when a user asks to archive, close out, or finalize an OpenSpec change after implementation and validation.

## Required context

1. Read `AGENTS.md`, `docs/agent_navigation.md`, `docs/maintainer_context_index.yaml`, and `docs/agent_context/openspec.md`.
2. Run `openspec list --json` and `openspec status --change <change> --json` to confirm whether the change is complete, active, or already archived.
3. Read the target change's proposal, design, tasks, and spec deltas before archiving. If tasks are incomplete, continue implementation or record a deferral instead of archiving.

## Workflow

1. Validate the target change with `openspec validate <change> --strict`.
2. Run any focused validation listed in the change tasks or scoped context.
3. Confirm that generated `outputs/`, `logs/`, cache, `dataset/` contents, and checkpoint files are not part of the source change.
4. Archive only a complete change. If the working tree shows both deleted active files and a new dated archive directory, report them as one paired closeout state.
5. After archiving, run `openspec validate --all --strict` and summarize the archive path and validation status.

## Commands and artifacts

- Run project Python commands through `conda run -n kd_mm_beam ...`.
- OpenSpec commands do not use the conda wrapper unless the local environment specifically requires it.
- Do not commit real `dataset/` contents, generated `outputs/`, `logs/`, cache, TensorBoard files, or new checkpoints.

```bash
openspec validate <change> --strict
openspec validate --all --strict
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
```
