---
name: kd-add-config
description: Add or revise KD-for-sensing configs, virtual recipes, overlays, and config migration guards without restoring retired routes.
license: MIT
---

# kd-add-config

Use this skill when adding or modifying YAML configs, canonical recipes, virtual config behavior, overlays, config migration guards, or config-facing documentation.

## Required context

1. Read `AGENTS.md`, `docs/agent_navigation.md`, `docs/maintainer_context_index.yaml`, and `docs/agent_context/configs.md`.
2. Read `openspec/specs/canonical-config-resolution/spec.md` and any workflow-specific OpenSpec spec.
3. If the config changes training flow, data contract, compatibility behavior, or public entrypoints, use an active OpenSpec change before editing.

## Workflow

1. Decide whether the config is canonical/current, virtual/generated, diagnostic, experiment reproduction, or local/manual.
2. Keep config semantics in `model.primary`, canonical recipe logic, or current owner modules; do not encode new runtime behavior only in docs.
3. Avoid retired tokens and paths. Virtual config must not take over old KD, HiST/Hist, residual, BGAM, viewer, Raymobtime, AMR-Net_gps_image, or JEPA-MSAC routes.
4. For Scene31 or other generator-backed families, keep generated YAML out of tracked source unless inventory and tests explicitly retain it.

## Commands and artifacts

- Run project Python commands through `conda run -n kd_mm_beam ...`.
- Do not commit real `dataset/` contents, generated `outputs/`, `logs/`, cache, or checkpoint files.
- Focused validation:

```bash
conda run -n kd_mm_beam pytest tests/test_config_load_characterization.py -q
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
make verify-cli-config
```
