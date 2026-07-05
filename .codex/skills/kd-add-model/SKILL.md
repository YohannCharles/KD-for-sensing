---
name: kd-add-model
description: Add or modify KD-for-sensing model components, baselines, registry entries, or model-facing configs while preserving OpenSpec, registry, and artifact boundaries.
license: MIT
---

# kd-add-model

Use this skill when adding or changing a model, forward contract, registry entry, baseline component, representation core, head, or model-facing config.

## Required context

1. Read `AGENTS.md`, `docs/agent_navigation.md`, `docs/maintainer_context_index.yaml`, and `docs/agent_context/models.md`.
2. Read the relevant OpenSpec specs, especially `openspec/specs/model-architecture-extension-contract/spec.md`, `openspec/specs/modular-sequence-model/spec.md`, and `openspec/specs/component-registry/spec.md`.
3. For non-trivial functionality, architecture, training-flow, data-contract, config-compatibility, or public-entrypoint changes, require an active OpenSpec change before code edits.

## Workflow

1. Classify the request as config-only baseline, component baseline, whole-model exception, or workflow/paper reproduction.
2. Prefer existing `modular_sequence` components and registry extension points before adding a whole model.
3. Put workflow reproduction logic under `src/kd_sensing/baselines/` or an existing owner module; do not copy the generic training loop.
4. Keep retired routes retired: old KD, HiST/Hist, Top8 selector, GPS residual, camera residual, BGAM, viewer manifest, AMR-Net_gps_image, and JEPA-MSAC cannot return as CLI, YAML, registry names, or wrappers.

## Commands and artifacts

- Run project Python commands through `conda run -n kd_mm_beam ...`.
- Do not commit real `dataset/` contents, generated `outputs/`, `logs/`, cache, TensorBoard files, or new checkpoint files.
- Focused validation usually starts with:

```bash
conda run -n kd_mm_beam pytest tests/test_component_registry.py -q
conda run -n kd_mm_beam pytest tests/test_architecture_boundaries.py -q
```

Add model, forward, objective, or runtime focused tests according to the changed owner.
