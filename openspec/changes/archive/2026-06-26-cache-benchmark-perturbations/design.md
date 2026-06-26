# Design: cache-benchmark-perturbations

## Approach
Keep this narrow: the benchmark already owns deterministic perturbation creation through `apply_difficulty_pipeline`. Add a small cache helper at that boundary.

The cache stores per-batch `torch.save` payloads:

- perturbed batch
- warnings
- suite id/type/condition/severity/seed/split
- sample ids and a cache schema version

The runner can use three modes:

- `off`: current online behavior
- `write`: apply perturbation online and write cache
- `read`: load cache, fail if missing
- `read_write`: load if present, otherwise generate and write

## Boundaries
- Do not create a new dataset format.
- Do not write cache files into source-controlled paths.
- Do not change labels, clean metrics, checkpoint loading, or model forward.
- Do not make cache mandatory for smoke tests or simple single-model runs.

## Validation
- Synthetic perturbation cache round-trip test.
- Existing JEPA GPS shortcut benchmark tests.
- OpenSpec strict validation.
