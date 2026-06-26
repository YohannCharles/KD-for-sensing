# Change: cache-benchmark-perturbations

## Why
JEPA GPS shortcut / predictive robustness real evaluation repeats the same deterministic image/GPS perturbations for every model. For multi-model comparisons this wastes CPU and data-loading time, and makes concurrent GPU jobs contend on identical preprocessing work.

## What Changes
- Add an opt-in benchmark perturbation cache that materializes deterministic perturbed evaluation batches once per suite/condition/severity/scene/split/seed.
- Let later benchmark runs read those cached perturbed batches and only run model forward + metric aggregation.
- Record cache provenance separately from model provenance so strict comparability still checks split, seed, sample ids and difficulty digest.

## Impact
- Scope: `src/kd_sensing/diagnostics/jepa_benchmark_*`, focused tests.
- Outputs: cache files live under ignored `outputs/cache/` or manifest-provided local output roots.
- Compatibility: default behavior remains online perturbation when cache is disabled or missing.
